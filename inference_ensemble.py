import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

import datetime
from pytz import timezone


def load_model(model_name, saved_model, num_classes, device):
    """
    저장된 모델의 가중치를 로드하는 함수입니다.

    Args:
        saved_model (str): 모델 가중치가 저장된 디렉토리 경로
        num_classes (int): 모델의 클래수 수
        device (torch.device): 모델이 로드될 장치 (CPU 또는 CUDA)

    Returns:
        model (nn.Module): 가중치가 로드된 모델
    """
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # 모델 가중치를 로드한다.
    model_path = os.path.join(saved_model, "best.pth")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


@torch.no_grad()
def inference(data_dir, model_dirs, output_dir, args):
    """
    모델 추론을 수행하는 함수

    Args:
        data_dir (str): 테스트 데이터가 있는 디렉토리 경로
        model_dir (str): 모델 가중치가 저장된 디렉토리 경로
        output_dir (str): 결과 CSV를 저장할 디렉토리 경로
        args (argparse.Namespace): 커맨드 라인 인자

    Returns:
        None
    """

    # CUDA를 사용할 수 있는지 확인
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    total_score = sum(args.scores)

    smax = nn.Softmax(dim=-1)

    preds_soft_mask = None
    preds_soft_gender = None
    preds_soft_age = None

    # 이미지 파일 경로와 정보 파일을 읽어온다.
    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    # 이미지 경로를 리스트로 생성한다.
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    for i in range(len(args.scores)):
        # 클래스의 개수를 설정한다. (마스크, 성별, 나이의 조합으로 18)
        num_classes = MaskBaseDataset.num_classes  # 18
        model = load_model(args.models[i], model_dirs[i], num_classes, device).to(device)
        model.eval()

        dataset = TestDataset(img_paths, args.resize[2 * i : 2 * i + 2])
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        if i == 0:
            preds_soft_mask = np.zeros((len(dataset), 3), dtype=np.float32)
            preds_soft_gender = np.zeros((len(dataset), 2), dtype=np.float32)
            preds_soft_age = np.zeros((len(dataset), 3), dtype=np.float32)

        print("Calculating inference results..")
        with torch.no_grad():
            for idx, images in enumerate(loader):
                # breakpoint()
                images = images.to(device)
                mask_output, gender_output, age_output = model(images)

                mask_output = smax(mask_output)
                mask_output = mask_output * args.scores[i] / total_score
                gender_output = smax(gender_output)
                gender_output = gender_output * args.scores[i] / total_score
                age_output = smax(age_output)
                age_output = age_output * args.scores[i] / total_score

                mask_output = mask_output.cpu().numpy()
                gender_output = gender_output.cpu().numpy()
                age_output = age_output.cpu().numpy()

                length = images.size(0)
                if length == args.batch_size:
                    preds_soft_mask[idx * length : (idx + 1) * length, :] += mask_output
                    preds_soft_gender[idx * length : (idx + 1) * length, :] += gender_output
                    preds_soft_age[idx * length : (idx + 1) * length, :] += age_output
                else:
                    preds_soft_mask[idx * args.batch_size : idx * args.batch_size + length, :] += mask_output
                    preds_soft_gender[
                        idx * args.batch_size : idx * args.batch_size + length, :
                    ] += gender_output
                    preds_soft_age[idx * args.batch_size : idx * args.batch_size + length, :] += age_output

    mask_preds = np.argmax(preds_soft_mask, axis=1)
    gender_preds = np.argmax(preds_soft_gender, axis=1)
    age_preds = np.argmax(preds_soft_age, axis=1)

    preds = mask_preds * 6 + gender_preds * 3 + age_preds

    # 예측 결과를 데이터프레임에 저장하고 csv 파일로 출력한다.
    info["ans"] = preds
    time_now = str(datetime.datetime.now(timezone("Asia/Seoul")))[:19]
    save_path = os.path.join(output_dir, f"output_{time_now}.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


# if __name__ == "__main__":
#     # 커맨드 라인 인자를 파싱한다.
#     parser = argparse.ArgumentParser()

#     # 데이터와 모델 체크포인트 디렉터리 관련 인자
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=500,
#         help="input batch size for validing (default: 500)",
#     )
#     parser.add_argument(
#         "--resize",
#         nargs=12,
#         type=int,
#         default=(128, 96, 128, 96, 128, 96, 128, 96, 128, 96, 128, 96),
#         help="resize size for image when you trained (default: (128, 96))",
#     )
#     parser.add_argument(
#         "--models",
#         nargs=6,
#         type=str,
#         default=("BaseModel", "BaseModel", "BaseModel", "BaseModel", "BaseModel", "BaseModel"),
#         help="model type (default: BaseModel x 6)",
#     )
#     parser.add_argument(
#         "--scores",
#         nargs=6,
#         type=float,
#         default=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
#         help="model type (default: BaseModel x 6)",
#     )

#     # 컨테이너 환경 변수
#     parser.add_argument(
#         "--data_dir",
#         type=str,
#         default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval"),
#     )
#     parser.add_argument(
#         "--model_dirs",
#         nargs=6,
#         type=str,
#         default=(
#             os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
#             os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
#             os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
#             os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
#             os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
#             os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
#         ),
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
#     )

#     args = parser.parse_args()

#     data_dir = args.data_dir
#     model_dirs = args.model_dirs
#     output_dir = args.output_dir

#     os.makedirs(output_dir, exist_ok=True)

#     # 모델 추론을 수행한다.
#     inference(data_dir, model_dirs, output_dir, args)

if __name__ == "__main__":
    # 커맨드 라인 인자를 파싱한다.
    parser = argparse.ArgumentParser()

    # 데이터와 모델 체크포인트 디렉터리 관련 인자
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="input batch size for validing (default: 500)",
    )
    parser.add_argument(
        "--resize",
        nargs=6,
        type=int,
        default=(128, 96, 128, 96, 128, 96),
        help="resize size for image when you trained (default: (128, 96))",
    )
    parser.add_argument(
        "--models",
        nargs=3,
        type=str,
        default=("BaseModel", "BaseModel", "BaseModel"),
        help="model type (default: BaseModel x 3)",
    )
    parser.add_argument(
        "--scores",
        nargs=3,
        type=float,
        default=(0.5, 0.5, 0.5),
        help="model type (default: BaseModel x 3)",
    )

    # 컨테이너 환경 변수
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval"),
    )
    parser.add_argument(
        "--model_dirs",
        nargs=3,
        type=str,
        default=(
            os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
            os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
            os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dirs = args.model_dirs
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # 모델 추론을 수행한다.
    inference(data_dir, model_dirs, output_dir, args)
