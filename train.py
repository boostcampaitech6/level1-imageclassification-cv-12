import argparse
import os
from importlib import import_module

import datetime
from pytz import timezone

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

#############################################################################################################################################
#
# python train.py --seed 6074 --epochs 30 --trainer Multi_coord_f --early_stopping 5 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation
# --resize 224 224 --batch_size 32 --valid_batch_size 500 --model EfficientNetViT --optimizer AdamW --criterion focal --lr_decay_step 10
# --name bkh_202312201600 --data_dir /data/ephemeral/home/removed_background --model_dir ./model/202312201600
#
# --name 은 각자의 이름 이니셜 들어가도록 허고 --data_dir 은 각자 서버에 있는 removed_background 로
#
#############################################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train (default: 1)")
    parser.add_argument(
        "--trainer",
        type=str,
        default="Single",
        help="trainer type Single or Multi (default: Single)",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=10,
        help="early stopping threshold (default: 10)",
    )
    parser.add_argument(
        "--k_limits",
        type=int,
        default=1,
        help="k for Stratified k-fold (default: 1)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskBaseDataset",
        help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="BaseAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[128, 96],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument("--model", type=str, default="BaseModel", help="model type (default: BaseModel)")
    parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer type (default: SGD)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}")

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/data/ephemeral/home/train/images"),
    )
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))

    parser.add_argument(
        "--resume_dir", type=str, default=None, help="path to latest checkpoint (default: None)"
    )

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    if args.k_limits > 1:
        trainer_name = "skf_" + args.trainer.lower()
    else:
        trainer_name = args.trainer.lower()
    train_module = getattr(import_module(f"trainer.{trainer_name}_trainer"), args.trainer + "Trainer")
    print(f" Use {trainer_name}_trainer ...")
    trainer = train_module(data_dir, model_dir, args)

    start_time = datetime.datetime.now(timezone("Asia/Seoul"))
    print(f"학습 시작 : {str(start_time)[:19]}")

    trainer.train(args)

    end_time = datetime.datetime.now(timezone("Asia/Seoul"))
    print(f"학습 끝 : {str(end_time)[:19]}")

    # 학습 소요 시간 계산 및 출력
    elapsed_time = end_time - start_time
    print(f"학습 소요 시간: {elapsed_time}")
