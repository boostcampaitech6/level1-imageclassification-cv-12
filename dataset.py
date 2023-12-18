import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
    CenterCrop,
    ColorJitter,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2

# https://github.com/mahmoudnafifi/WB_color_augmenter
# https://colab.research.google.com/drive/1wbUW87MoXabdzDh53YWoOXrvOdjpubQ4?usp=sharing#scrollTo=T226o1Jb64P3
# Mahmoud Afifi and Michael S. Brown. What Else Can Fool Deep Learning? Addressing Color Constancy Errors on Deep Neural Network Performance. International Conference on Computer Vision (ICCV), 2019.
from WBAugmenter import WBEmulator as wbAug
from tqdm import tqdm
import pickle


# 지원되는 이미지 확장자 리스트
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    """파일 이름이 이미지 확장자를 가지는지 확인하는 함수

    Args:
        filename (str): 확인하고자 하는 파일 이름

    Returns:
        bool: 파일 이름이 이미지 확장자를 가지면 True, 그렇지 않으면 False.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    """
    기본적인 Augmentation을 담당하는 클래스

    Attributes:
        transform (Compose): 이미지를 변환을 위한 torchvision.transforms.Compose 객체
    """

    def __init__(self, resize, mean, std, **args):
        """
        Args:
            resize (tuple): 이미지의 리사이즈 대상 크지
            mean (tuple): Normalize 변환을 위한 평균 값
            std (tuple): Normalize 변환을 위한 표준 값
        """
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        """
        이미지에 저장된 transform 적용

        Args:
            Image (PIL.Image): Augumentation을 적용할 이미지

        Returns:
            Tensor: Argumentation이 적용된 이미지
        """
        return self.transform(image)


class AddGaussianNoise(object):
    """이미지에 Gaussian Noise를 추가하는 클래스"""

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class CustomAugmentation:

    """커스텀 Augmentation을 담당하는 클래스 -> albumentations 사용"""

    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose(
            [
                A.CenterCrop(height=320, width=256),
                A.Resize(*resize, interpolation=0),
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.3, 0.3), p=0.5),
                A.GaussNoise(),
                A.CoarseDropout(
                    max_holes=10,
                    max_height=8,
                    max_width=8,
                    min_holes=None,
                    min_height=5,
                    min_width=5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __call__(self, img):
        augmented = self.transform(image=np.array(img).astype(np.uint8))

        return augmented["image"]


class MaskLabels(int, Enum):
    """마스크 라벨을 나타내는 Enum 클래스"""

    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    """성별 라벨을 나타내는 Enum 클래스"""

    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        """문자열로부터 해당하는 성별 라벨을 찾아 반환하는 클래스 메서드"""
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    """나이 라벨을 나타내는 Enum 클래스"""

    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        """숫자로부터 해당하는 나이 라벨을 찾아 반환하는 클래스 메서드"""
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    """마스크 데이터셋의 기본 클래스"""

    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        aug_prob=0.5,
    ):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()  # 통계시 계산 (평균 및 표준 편차)
        self.wb_color_aug = wbAug.WBEmulator()
        self.mapping = self.compute_mapping()
        self.aug_prob = aug_prob

    def compute_mapping(self):
        """
        White balance를 적용할 때 사용할 mapping을 만들고 불러오는 함수입니다

        Returns:
            List: white balance mapping을 담은 List를 반환합니다
        """
        temp = os.path.split(self.data_dir)[0]
        if os.path.exists(os.path.join(temp, "wb_mfs.pickle")):
            with open(os.path.join(temp, "wb_mfs.pickle"), "rb") as handle:
                mapping_funcs = pickle.load(handle)
            return mapping_funcs

        print("Computing mapping functions for WB augmenter. " "This process may take time....")
        mapping_funcs = []
        for idx in tqdm(range(self.__len__())):
            img = self.read_image(idx)
            mfs = self.wb_color_aug.computeMappingFunc(img)
            mapping_funcs.append(mfs)
        with open(os.path.join(temp, "wb_mfs.pickle"), "wb") as handle:
            pickle.dump(mapping_funcs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return mapping_funcs

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith(".") or profile.endswith('pickle'):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        if random.random() < self.aug_prob:
            mfs = self.mapping[index]
            ind = np.random.randint(len(mfs))
            mf = mfs[ind]
            image = wbAug.changeWB(np.array(image), mf)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        """인덱스에 해당하는 마스크 라벨을 반환하는 메서드"""
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        """인덱스에 해당하는 성별 라벨을 반환하는 메서드"""
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        """인덱스에 해당하는 나이 라벨을 반환하는 메서드"""
        return self.age_labels[index]

    def read_image(self, index):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        """다중 라벨을 하나의 클래스로 인코딩하는 메서드"""
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        """인코딩된 다중 라벨을 각각의 라벨로 디코딩하는 메서드"""
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
    train / val 나누는 기준을 이미지에 대해서 random 이 아닌 사람(profile)을 기준으로 나눕니다.
    구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다.
    이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        aug_prob=0.5,
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio, aug_prob)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        """프로필을 학습과 검증용으로 나누는 메서드"""
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def setup(self):
        """데이터셋 설정을 하는 메서드. 프로필 기준으로 나눈다."""
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                if img_folder.endswith('pickle'):
                    continue
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        """프로필 기준으로 나눈 데이터셋을 Subset 리스트로 반환하는 메서드"""
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class BalancedDataset(MaskSplitByProfileDataset):
    """
    MaskSplitByProfileDataset 처럼 사람을 기준으로 train/val을 나누고 있습니다.
    갯수가 적은 클래스의 사진들을 단순 중복 입력하여 클래스 간 불균형을 완화하려합니다.
    """

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        aug_prob=0.5,
    ):
        super().__init__(data_dir, mean, std, val_ratio, aug_prob)

    def compute_mapping(self):
        """
        MaskBaseDataset의 compute_mapping 함수를 override 합니다
        데이터 갯수가 늘어난 상태에서 기존의 wb_mfs.pickle 파일을 사용하면 에러가 발생하므로
        wb_mfs_c.pickle를 새로 생성합니다

        Returns:
            List: white balance mapping을 담은 List를 반환합니다
        """
        temp = os.path.split(self.data_dir)[0]
        if os.path.exists(os.path.join(temp, "wb_mfs_c.pickle")):
            with open(os.path.join(temp, "wb_mfs_c.pickle"), "rb") as handle:
                mapping_funcs = pickle.load(handle)
            return mapping_funcs

        print("Computing mapping functions for WB augmenter. " "This process may take time....")
        mapping_funcs = []
        for idx in tqdm(range(self.__len__())):
            img = self.read_image(idx)
            mfs = self.wb_color_aug.computeMappingFunc(img)
            mapping_funcs.append(mfs)
        with open(os.path.join(temp, "wb_mfs_c.pickle"), "wb") as handle:
            pickle.dump(mapping_funcs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return mapping_funcs

    def setup(self):
        """
        MaskSplitByProfileDataset의 setup 함수를 override합니다
        갯수가 적은 클래스의 사진들을 데이터셋에 단순 중복 입력하고 있습니다
        """
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)
        # split_profiles 는 {"train": train_indices, "val": val_indices} 형태

        cnt = 0
        for phase, indices in split_profiles.items():
            # phase는 'train' 또는 'val'
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                if img_folder.endswith('pickle'):
                    continue
                for file_name in os.listdir(img_folder):
                    # file_name은 'incorrect_mask.jpg' 'mask4.jpg' 이런 형태
                    _file_name, ext = os.path.splitext(file_name)
                    # 파일 확장자 분리
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]
                    # mask 착용이면 0, 이상착용이면 1, 미착용이면 2

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    # 남자 0, 여자 1
                    age_label = AgeLabels.from_number(age)
                    # ~29 이면 0, 30~59 이면 1, 60이상이면 2

                    loop_mask = 1
                    if mask_label != 0:
                        loop_mask == 5
                        # 마스크를 안쓴 경우, 마스크를 부적절하게 쓴 경우인 사진들을 5번 중복 입력합니다

                    loop_gender = 1
                    if gender_label == 0:
                        loop_gender = 2
                        # 남성 사진을 2번 중복 입력합니다

                    loop_age = 1
                    if age_label == 2 and gender_label == 1:
                        loop_age = 7
                        # 여성 60대 이상을 7번 중복 입력합니다

                    if age_label == 2 and gender_label == 0:
                        loop_age = 5
                        # 남성 60대 이상을 5번 중복 입력합니다

                    for i in range(loop_mask):
                        for j in range(loop_gender):
                            for k in range(loop_age):
                                # 이전 단계에서 결정된 중복 횟수들을 모두 곱한 횟수만큼 중복 입력합니다
                                self.image_paths.append(img_path)
                                self.mask_labels.append(mask_label)
                                self.gender_labels.append(gender_label)
                                self.age_labels.append(age_label)

                                self.indices[phase].append(cnt)
                                cnt += 1


class TestDataset(Dataset):
    """테스트 데이터셋 클래스"""

    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.img_paths)
