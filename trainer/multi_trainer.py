import glob
import json
import multiprocessing
import os
import random
import re
import wandb
from importlib import import_module
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


class MultiTrainer:
    """
    1. single classifier train
    2. wandb 연결
    3. early stopping 구현 - args.early_stopping 로 threshold 설정
    4. Stratified k-fold 구현 - args.k_limits 로 분할 횟수 설정
    5. WeightedRandomSampler 구현
    """
    
    def __init__(self, data_dir, model_dir, **args):
        """
        Args:
            args.seed: random seed (default: 42)
            args.epochs: number of epochs to train (default: 1)
            args.early_stopping: early stopping threshold (default: 10)
            args.k_limits: k for Stratified k-fold (default: 1)
            ...
        """
        
        self.data_dir = data_dir
        self.model_dr = model_dir
        
        wandb.init(
        project="Boostcamp_Mask_ImageClassification",
        notes="",
        config={
                "Architecture": args.model,
                "Img_size": args.resize,
                "Loss": args.criterion,
                "Learning_rate": args.lr,
                "Epochs": args.epochs,
            },
        )
        wandb.run.name = args.name
        wandb.run.save()

    def seed_everything(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)


    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]


    def grid_image(self, np_images, gts, preds, n=16, shuffle=False):
        batch_size = np_images.shape[0]
        assert n <= batch_size

        choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
        figure = plt.figure(
            figsize=(12, 18 + 2)
        )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
        plt.subplots_adjust(
            top=0.8
        )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
        n_grid = int(np.ceil(n**0.5))
        tasks = ["mask", "gender", "age"]
        for idx, choice in enumerate(choices):
            gt = gts[choice].item()
            pred = preds[choice].item()
            image = np_images[choice]
            gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
            pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
            title = "\n".join(
                [
                    f"{task} - gt: {gt_label}, pred: {pred_label}"
                    for gt_label, pred_label, task in zip(
                        gt_decoded_labels, pred_decoded_labels, tasks
                    )
                ]
            )

            plt.subplot(n_grid, n_grid, idx + 1, title=title)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image, cmap=plt.cm.binary)

        return figure


    def increment_path(self, path, exist_ok=False):
        """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

        Args:
            path (str or pathlib.Path): f"{model_dir}/{args.name}".
            exist_ok (bool): whether increment path (increment if False).
        """
        path = Path(path)
        if (path.exists() and exist_ok) or (not path.exists()):
            return str(path)
        else:
            dirs = glob.glob(f"{path}*")
            matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 2
            return f"{path}{n}"

    # -- Focal Loss 를 위한 가중치 계산
    def compute_class_weights(self, labels):
        """
        Focal Loss 를 위한 가중치 계산 함수 입니다.
        [백광현]
        """
        class_counts = torch.bincount(labels)
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        return class_weights
    
    
    def train(self, args):
        
        self.seed_everything(args.seed)
        save_dir = self.increment_path(os.path.join(self.model_dir, args.name))

        # -- settings
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # -- early stopping flag
        patience = args.early_stopping
        counter = 0
        
        # -- dataset
        dataset_module = getattr(
            import_module("dataset"), args.dataset
        )  # default: MaskBaseDataset
        dataset = dataset_module(
            data_dir=self.data_dir,
        )
        num_classes = dataset.num_classes  # 18

        # -- augmentation
        transform_module = getattr(
            import_module("dataset"), args.augmentation
        )  # default: BaseAugmentation
        transform = transform_module(
            resize=args.resize,
            mean=dataset.mean,
            std=dataset.std,
        )
        dataset.set_transform(transform)

        # -- weightedRandomSampler
        train_set, val_set = dataset.split_dataset()
        
        labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]
        class_counts = Counter(labels)
        total_samples = len(labels)
        indices = train_set.indices
        class_weights = {class_label: total_samples / count for class_label, count in class_counts.items()}
        weights = [class_weights[labels[i]] for i in indices]
        
        # -- data_loader
        sampler = WeightedRandomSampler(weights=torch.Tensor(weights), num_samples=len(train_set), replacement=True)

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            sampler=sampler,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )
        
        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(num_classes=num_classes).to(device)
        train_params = [{'params': getattr(model, 'features').parameters(), 'lr': args.lr / 10, 'weight_decay':5e-4},
                    {'params': getattr(model, 'mask_classifier').parameters(), 'lr': args.lr, 'weight_decay':5e-4},
                    {'params': getattr(model, 'gender_classifier').parameters(), 'lr': args.lr, 'weight_decay':5e-4},
                    {'params': getattr(model, 'age_classifier').parameters(), 'lr': args.lr, 'weight_decay':5e-4}]
        
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        m_cls_weight = self.compute_class_weights(torch.tensor(dataset.mask_labels, device=device))
        g_cls_weight = self.compute_class_weights(torch.tensor(dataset.gender_labels, device=device))
        a_cls_weight = self.compute_class_weights(torch.tensor(dataset.age_labels, device=device))
        
        if args.criterion == 'focal':
            m_criterion = create_criterion(args.criterion, alpha=m_cls_weight)
            g_criterion = create_criterion(args.criterion, alpha=g_cls_weight)
            a_criterion = create_criterion(args.criterion, alpha=a_cls_weight)
        else:
            criterion = create_criterion(args.criterion)
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(train_params)
        # optimizer = opt_module(
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     lr=args.lr,
        #     weight_decay=5e-4,
        # )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            
            loss_value = 0
            m_value = 0
            g_value = 0
            a_value = 0
            
            matches = 0
            mask_matches = 0
            gender_matches = 0
            age_matches = 0
            
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                mask_label, gender_label, age_label = dataset.decode_multi_class(labels)

                optimizer.zero_grad()

                mask_output, gender_output, age_output = model(inputs)
                
                if args.criterion == 'focal':
                    mask_loss = m_criterion(mask_output, mask_label)
                    gender_loss = g_criterion(gender_output, gender_label)
                    age_loss = a_criterion(age_output, age_label)
                else:
                    mask_loss = criterion(mask_output, mask_label)
                    gender_loss = criterion(gender_output, gender_label)
                    age_loss = criterion(age_output, age_label)
                
                # mask_loss.backward(retain_graph=True)
                # gender_loss.backward(retain_graph=True)
                # age_loss.backward()
                
                sum_loss = mask_loss + gender_loss + 1.5 * age_loss
                sum_loss.backward()
                
                mask_pred = torch.argmax(mask_output, dim=-1)
                gender_pred = torch.argmax(gender_output, dim=-1)
                age_pred = torch.argmax(age_output, dim=-1)
                preds = mask_pred * 6 + gender_pred * 3 + age_pred
                #loss = criterion(outs, labels)
                #loss.backward()
                
                optimizer.step()

                loss_value += sum_loss.item()
                m_value += mask_loss.item()
                g_value += gender_loss.item()
                a_value += age_loss.item()
                
                matches += (preds == labels).sum().item()
                mask_matches += (mask_pred == mask_label).sum().item()
                gender_matches += (gender_pred == gender_label).sum().item()
                age_matches += (age_pred == age_label).sum().item()
                
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    m_loss = m_value / args.log_interval
                    g_loss = g_value / args.log_interval
                    a_loss = a_value / args.log_interval
                    
                    train_acc = matches / args.batch_size / args.log_interval
                    m_acc = mask_matches / args.batch_size / args.log_interval
                    g_acc = gender_matches / args.batch_size / args.log_interval
                    a_acc = age_matches / args.batch_size / args.log_interval
                    current_lr = self.get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} m_loss {m_loss:4.4} g_loss {g_loss:4.4} a_loss {a_loss:4.4} || training accuracy {train_acc:4.2%} m_acc {m_acc:4.2%} g_acc {g_acc:4.2%} a_acc {a_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar(
                        "Train/loss", train_loss, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                    )

                    loss_value = 0
                    m_value = 0
                    g_value = 0
                    a_value = 0
                    
                    matches = 0
                    mask_matches = 0
                    gender_matches = 0
                    age_matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                val_f1_items = []  # 추가: F1 score를 기록하기 위한 리스트
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    mask_label, gender_label, age_label = dataset.decode_multi_class(labels)

                    #outs = model(inputs)
                    mask_output, gender_output, age_output = model(inputs)
                    mask_pred = torch.argmax(mask_output, dim=-1)
                    gender_pred = torch.argmax(gender_output, dim=-1)
                    age_pred = torch.argmax(age_output, dim=-1)
                    preds = mask_pred * 6 + gender_pred * 3 + age_pred
                    #preds = torch.argmax(outs, dim=-1)
                    
                    if args.criterion == 'focal':
                        mask_loss = m_criterion(mask_output, mask_label)
                        gender_loss = g_criterion(gender_output, gender_label)
                        age_loss = a_criterion(age_output, age_label)
                    else:
                        mask_loss = criterion(mask_output, mask_label)
                        gender_loss = criterion(gender_output, gender_label)
                        age_loss = criterion(age_output, age_label)
                    sum_loss = mask_loss + gender_loss + 1.5 * age_loss
                    
                    loss_item = sum_loss.item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    
                    # F1 score 계산
                    f1_item = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    val_f1_items.append(f1_item)

                    if figure is None:
                        inputs_np = (
                            torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        )
                        inputs_np = dataset_module.denormalize_image(
                            inputs_np, dataset.mean, dataset.std
                        )
                        figure = self.grid_image(
                            inputs_np,
                            labels,
                            preds,
                            n=16,
                            shuffle=args.dataset != "MaskSplitByProfileDataset",
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                val_f1 = np.mean(val_f1_items)  # 추가: F1 score의 평균을 계산
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(
                        f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                    )
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_acc = val_acc
                    counter = 0
                else:
                    counter += 1
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, F1 score: {val_f1:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_scalar("Val/f1_score", val_f1, epoch)  # 추가: F1 score를 기록
                logger.add_figure("results", figure, epoch)
                
                wandb.log(
                    {
                        "Train Loss": train_loss,
                        "Train Accuracy": train_acc,
                        "Val Loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Val F1 Score": val_f1
                    }
                )
                
                print()
                
                if counter > patience:
                    print("Early Stopping...")
                    break
