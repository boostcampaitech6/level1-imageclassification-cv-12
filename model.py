import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import *

import timm


class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# -- mask, gender, age 따로따로 classifier 제작, 단 feature 는 공유(backbone 1개)
class MultiLabelModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelModel, self).__init__()

        """
        1. backbone 선택 후 classifier 차원 수 설정
        2. 모델의 output_dimension 은 num_classes 로 설정
        """

        mask_num_classes = int(num_classes // 6)
        gender_num_classes = int(num_classes // 9)
        age_num_classes = int(num_classes // 6)

        # pretrained model -> 각 모델의 마지막 fc layer 를 빼고 차원 수 맞춰주는 작업 필요
        efficientnet = efficientnet_b7(pretrained=True)
        self.features = nn.Sequential(*list(efficientnet.children())[:-1])

        # # Freeze pretrained weights
        # for param in self.features.parameters():
        #     param.requires_grad = False

        # three classifier
        self.mask_classifier = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, mask_num_classes),
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, gender_num_classes),
        )

        self.age_classifier = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, age_num_classes),
        )

        self.initialize_weights(self.mask_classifier)
        self.initialize_weights(self.gender_classifier)
        self.initialize_weights(self.age_classifier)

    def forward(self, x):
        """
        1. 클래스 별로 3개의 output 출력
        """
        # Feature extraction
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten features

        # Task-specific output
        mask_output = self.mask_classifier(features)
        gender_output = self.gender_classifier(features)
        age_output = self.age_classifier(features)

        return mask_output, gender_output, age_output

    def initialize_weights(self, model):
        """
        He 가중치 초기화
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                m.bias.data.zero_()


# -- backbone 을 나눠서 ResNet50 2개를 사용
class MFResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. backbone 선택 후 classifier 차원 수 설정
        2. 모델의 output_dimension 은 num_classes 로 설정
        """

        mask_num_classes = int(num_classes // 6)
        gender_num_classes = int(num_classes // 9)
        age_num_classes = int(num_classes // 6)

        # pretrained model -> 각 모델의 마지막 fc layer 를 빼고 차원 수 맞춰주는 작업 필요
        self.backbone1 = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.backbone2 = timm.create_model("resnet50", pretrained=True, num_classes=0)

        self.mask_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, mask_num_classes),
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, gender_num_classes),
        )

        self.age_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, age_num_classes),
        )

        self.initialize_weights(self.mask_classifier)
        self.initialize_weights(self.gender_classifier)
        self.initialize_weights(self.age_classifier)

    def forward(self, x):
        """
        1. Mask와 Gender 는 feature 를 공유하고, Age 는 따로 feature를 사용하여 클래스 별로 3개의 output 출력
        2. feature는 각각 Mask / Gender, Age
        """
        # Feature extraction
        m_features = self.backbone1(x)
        m_features = m_features.view(m_features.size(0), -1)  # Flatten features

        ga_features = self.backbone2(x)
        ga_features = ga_features.view(ga_features.size(0), -1)

        # Task-specific & Multi feature output
        mask_output = self.mask_classifier(m_features)

        gender_output = self.gender_classifier(ga_features)
        age_output = self.age_classifier(ga_features)

        return mask_output, gender_output, age_output

    def initialize_weights(self, model):
        """
        He 가중치 초기화
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                m.bias.data.zero_()


# -- EfficientNet_b7 2개 사용
class MFEfficient(nn.Module):
    def __init__(self, num_classes):
        super(MFEfficientResNet, self).__init__()

        """
        1. backbone 선택 후 classifier 차원 수 설정
        2. 모델의 output_dimension 은 num_classes 로 설정
        """

        mask_num_classes = int(num_classes // 6)
        gender_num_classes = int(num_classes // 9)
        age_num_classes = int(num_classes // 6)

        # pretrained model -> 각 모델의 마지막 fc layer 를 빼고 차원 수 맞춰주는 작업 필요
        efficientnet1 = efficientnet_b7(pretrained=True)
        efficientnet2 = efficientnet_b7(pretrained=True)
        self.backbone1 = nn.Sequential(*list(efficientnet1.children())[:-1])
        self.backbone2 = nn.Sequential(*list(efficientnet2.children())[:-1])

        # # Freeze pretrained weights
        # for param in self.features.parameters():
        #     param.requires_grad = False

        self.mask_classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, mask_num_classes),
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, gender_num_classes),
        )

        self.age_classifier = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, age_num_classes),
        )

        self.initialize_weights(self.mask_classifier)
        self.initialize_weights(self.gender_classifier)
        self.initialize_weights(self.age_classifier)

    def forward(self, x):
        """
        1. Mask와 Gender 는 feature 를 공유하고, Age 는 따로 feature를 사용하여 클래스 별로 3개의 output 출력
        2. feature는 각각 Mask / Gender, Age
        """
        # Feature extraction
        m_features = self.backbone1(x)
        m_features = m_features.view(m_features.size(0), -1)  # Flatten features

        ga_features = self.backbone2(x)
        ga_features = ga_features.view(ga_features.size(0), -1)

        # Task-specific & Multi feature output
        mask_output = self.mask_classifier(m_features)

        gender_output = self.gender_classifier(ga_features)
        age_output = self.age_classifier(ga_features)

        return mask_output, gender_output, age_output

    def initialize_weights(self, model):
        """
        He 가중치 초기화
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                m.bias.data.zero_()


# -- EfficientNet, ResNet 혼용
class MFEfficientResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. backbone 선택 후 classifier 차원 수 설정
        2. 모델의 output_dimension 은 num_classes 로 설정
        """

        mask_num_classes = int(num_classes // 6)
        gender_num_classes = int(num_classes // 9)
        age_num_classes = int(num_classes // 6)

        # pretrained model -> 각 모델의 마지막 fc layer 를 빼고 차원 수 맞춰주는 작업 필요
        resnet = resnet101(pretrained=True)
        efficientnet = efficientnet_b7(pretrained=True)
        self.backbone1 = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone2 = nn.Sequential(*list(efficientnet.children())[:-1])

        # # Freeze pretrained weights
        # for param in self.features.parameters():
        #     param.requires_grad = False

        self.mask_classifier = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, mask_num_classes),
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, gender_num_classes),
        )

        self.age_classifier = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, age_num_classes),
        )

        self.initialize_weights(self.mask_classifier)
        self.initialize_weights(self.gender_classifier)
        self.initialize_weights(self.age_classifier)

    def forward(self, x):
        """
        1. Mask와 Gender 는 feature 를 공유하고, Age 는 따로 feature를 사용하여 클래스 별로 3개의 output 출력
        2. feature는 각각 Mask / Gender, Age
        """
        # Feature extraction
        m_features = self.backbone1(x)
        m_features = m_features.view(m_features.size(0), -1)  # Flatten features

        ga_features = self.backbone2(x)
        ga_features = ga_features.view(ga_features.size(0), -1)

        # Task-specific & Multi feature output
        mask_output = self.mask_classifier(m_features)

        gender_output = self.gender_classifier(ga_features)
        age_output = self.age_classifier(ga_features)

        return mask_output, gender_output, age_output

    def initialize_weights(self, model):
        """
        He 가중치 초기화
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                m.bias.data.zero_()


# -- Mixer 사용
class Mixer(nn.Module):
    def __init__(self, num_classes):
        super(Mixer, self).__init__()

        """
        1. backbone 선택 후 classifier 차원 수 설정
        2. 모델의 output_dimension 은 num_classes 로 설정
        """

        mask_num_classes = int(num_classes // 6)
        gender_num_classes = int(num_classes // 9)
        age_num_classes = int(num_classes // 6)

        # pretrained model -> 각 모델의 마지막 fc layer 를 빼고 차원 수 맞춰주는 작업 필요
        mixer1 = timm.create_model("mixer_b16_224_miil_in21k", pretrained=True)
        mixer2 = timm.create_model("mixer_b16_224_miil_in21k", pretrained=True)
        self.backbone1 = nn.Sequential(*list(mixer1.children())[:-1])
        self.backbone2 = nn.Sequential(*list(mixer2.children())[:-1])

        # # Freeze pretrained weights
        # for param in self.features.parameters():
        #     param.requires_grad = False

        self.mask_classifier = nn.Sequential(
            nn.Linear(150528, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, mask_num_classes),
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(150528, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, gender_num_classes),
        )

        self.age_classifier = nn.Sequential(
            nn.Linear(150528, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, age_num_classes),
        )

        self.initialize_weights(self.mask_classifier)
        self.initialize_weights(self.gender_classifier)
        self.initialize_weights(self.age_classifier)

    def forward(self, x):
        """
        1. Mask와 Gender 는 feature 를 공유하고, Age 는 따로 feature를 사용하여 클래스 별로 3개의 output 출력
        """
        # Feature extraction
        m_features = self.backbone1(x)
        m_features = m_features.view(m_features.size(0), -1)  # Flatten features

        ga_features = self.backbone2(x)
        ga_features = ga_features.view(ga_features.size(0), -1)

        # Task-specific & Multi feature output
        mask_output = self.mask_classifier(m_features)

        gender_output = self.gender_classifier(ga_features)
        age_output = self.age_classifier(ga_features)

        return mask_output, gender_output, age_output

    def initialize_weights(self, model):
        """
        He 가중치 초기화
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                m.bias.data.zero_()


# -- EfficientNet 과 EfficientViT 혼용
class EfficientNetViT(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetViT, self).__init__()

        """
        1. backbone 선택 후 classifier 차원 수 설정
        2. 모델의 output_dimension 은 num_classes 로 설정
        """

        mask_num_classes = int(num_classes // 6)
        gender_num_classes = int(num_classes // 9)
        age_num_classes = int(num_classes // 6)

        # pretrained model -> 각 모델의 마지막 fc layer 를 빼고 차원 수 맞춰주는 작업 필요
        effvit = timm.create_model(
            "efficientvit_b3.r224_in1k", pretrained=True, num_classes=0
        )
        efficientnet = efficientnet_b5(pretrained=True)
        self.backbone1 = nn.Sequential(*list(efficientnet.children())[:-1])
        self.backbone2 = effvit

        # # Freeze pretrained weights
        # for param in self.features.parameters():
        #     param.requires_grad = Falsev

        # three classifier
        self.mask_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, mask_num_classes),
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, gender_num_classes),
        )

        self.age_classifier = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm((256,), 1e-05, elementwise_affine=True),
            nn.Hardswish(),
            nn.Dropout(p=0.0, inplace=False),
            nn.Linear(256, age_num_classes, bias=True),
        )

        self.initialize_weights(self.mask_classifier)
        self.initialize_weights(self.gender_classifier)
        self.initialize_weights(self.age_classifier)

    def forward(self, x):
        """
        1. Mask와 Gender 는 feature 를 공유하고, Age 는 따로 feature를 사용하여 클래스 별로 3개의 output 출력
        """
        # Feature extraction
        mg_features = self.backbone1(x)
        mg_features = mg_features.view(mg_features.size(0), -1)  # Flatten features

        a_features = self.backbone2(x)
        # ga_features = ga_features.view(ga_features.size(0), -1)

        # Task-specific & Multi feature output
        mask_output = self.mask_classifier(mg_features)
        gender_output = self.gender_classifier(mg_features)

        age_output = self.age_classifier(a_features)

        return mask_output, gender_output, age_output

    # def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
    #     if x.dim() > 2:
    #         x = torch.flatten(x, start_dim=1)
    #     return x

    def initialize_weights(self, model):
        """
        He 가중치 초기화
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EfficientNetViT_b4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. backbone 선택 후 classifier 차원 수 설정
        2. 모델의 output_dimension 은 num_classes 로 설정
        """

        mask_num_classes = int(num_classes // 6)
        gender_num_classes = int(num_classes // 9)
        age_num_classes = int(num_classes // 6)

        # pretrained model -> 각 모델의 마지막 fc layer 를 빼고 차원 수 맞춰주는 작업 필요
        effvit = timm.create_model(
            "efficientvit_b3.r256_in1k", pretrained=True, num_classes=0
        )
        efficientnet = efficientnet_b4(pretrained=True)
        self.backbone1 = nn.Sequential(*list(efficientnet.children())[:-1])
        self.backbone2 = effvit

        # # Freeze pretrained weights
        # for param in self.features.parameters():
        #     param.requires_grad = Falsev

        # three classifier
        self.mask_classifier = nn.Sequential(
            nn.Linear(1792, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, mask_num_classes),
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(1792, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(512, gender_num_classes),
        )

        self.age_classifier = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm((256,), 1e-05, elementwise_affine=True),
            nn.Hardswish(),
            nn.Dropout(p=0.0, inplace=False),
            nn.Linear(256, age_num_classes, bias=True),
        )

        self.initialize_weights(self.mask_classifier)
        self.initialize_weights(self.gender_classifier)
        self.initialize_weights(self.age_classifier)

    def forward(self, x):
        """
        1. Mask와 Gender 는 feature 를 공유하고, Age 는 따로 feature를 사용하여 클래스 별로 3개의 output 출력
        """
        # Feature extraction
        mg_features = self.backbone1(x)
        mg_features = mg_features.view(mg_features.size(0), -1)  # Flatten features

        a_features = self.backbone2(x)
        # ga_features = ga_features.view(ga_features.size(0), -1)

        # Task-specific & Multi feature output
        mask_output = self.mask_classifier(mg_features)
        gender_output = self.gender_classifier(mg_features)

        age_output = self.age_classifier(a_features)

        return mask_output, gender_output, age_output

    # def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
    #     if x.dim() > 2:
    #         x = torch.flatten(x, start_dim=1)
    #     return x

    def initialize_weights(self, model):
        """
        He 가중치 초기화
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# -- EfficientViT 2개 혼용
class DualEfficientViT(nn.Module):
    def __init__(self, num_classes):
        super(DualEfficientViT, self).__init__()

        """
        1. backbone 선택 후 classifier 차원 수 설정
        2. 모델의 output_dimension 은 num_classes 로 설정
        """

        mask_num_classes = int(num_classes // 6)
        gender_num_classes = int(num_classes // 9)
        age_num_classes = int(num_classes // 6)

        # pretrained model -> 각 모델의 마지막 fc layer 를 빼고 차원 수 맞춰주는 작업 필요
        effvit1 = timm.create_model(
            "efficientvit_b3.r224_in1k", pretrained=True, num_classes=0
        )
        effvit2 = timm.create_model(
            "efficientvit_b3.r224_in1k", pretrained=True, num_classes=0
        )
        self.backbone1 = effvit1
        self.backbone2 = effvit2

        # # Freeze pretrained weights
        # for param in self.features.parameters():
        #     param.requires_grad = Falsev

        # three classifier
        self.mask_classifier = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm((256,), 1e-05, elementwise_affine=True),
            nn.Hardswish(),
            nn.Dropout(p=0.0, inplace=False),
            nn.Linear(256, mask_num_classes, bias=True),
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm((256,), 1e-05, elementwise_affine=True),
            nn.Hardswish(),
            nn.Dropout(p=0.0, inplace=False),
            nn.Linear(256, gender_num_classes, bias=True),
        )

        self.age_classifier = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm((256,), 1e-05, elementwise_affine=True),
            nn.Hardswish(),
            nn.Dropout(p=0.0, inplace=False),
            nn.Linear(256, age_num_classes, bias=True),
        )

        self.initialize_weights(self.mask_classifier)
        self.initialize_weights(self.gender_classifier)
        self.initialize_weights(self.age_classifier)

    def forward(self, x):
        """
        1. Mask와 Gender 는 feature 를 공유하고, Age 는 따로 feature를 사용하여 클래스 별로 3개의 output 출력
        """
        # Feature extraction
        mg_features = self.backbone1(x)
        # mg_features = mg_features.view(mg_features.size(0), -1)  # Flatten features

        a_features = self.backbone2(x)
        # ga_features = ga_features.view(ga_features.size(0), -1)

        # Task-specific & Multi feature output
        mask_output = self.mask_classifier(mg_features)
        gender_output = self.gender_classifier(mg_features)

        age_output = self.age_classifier(a_features)

        return mask_output, gender_output, age_output

    # def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
    #     if x.dim() > 2:
    #         x = torch.flatten(x, start_dim=1)
    #     return x

    def initialize_weights(self, model):
        """
        He 가중치 초기화
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
