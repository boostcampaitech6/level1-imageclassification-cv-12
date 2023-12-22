# LEVEL 1 IMAGE CLASSIFICATION
2023.12.11 ~ 2023.12.22

## CV_12
김시웅, 이동형, 조형서, 백광현, 박정민
- - -
### Paths
```
├── data
│   ├── train
│   │   ├── images
│   │   └── train.csv
│   │
│   └── eval
│       ├── images
│       └── info.csv
│
├── trainer
│   ├── utils
│   │   ├── __init__.py
│   │   ├── aws_s3_downloader.py
│   │   └── utils.py
│   │
│   ├── __init__.py
│   └── pytorch_kobert.py
│
├── dataset.py
├── loss.py
├── model.py
├── preprocess.py
├── inference.py
├── inference_ensemble.py
├── requirements.txt
└── README.md
```
- - -
### 프로젝트 개요
COVID-19 바이러스는 주로 입과 호흡기에서 나오는 비말을 통해 전파되므로, 모든 사람이 마스크를 착용하여 전파 경로를 차단하는 것이 중요하다. 특히 공공장소에서의 마스크 착용은 필수적이며, 코와 입을 완전히 가리는 올바른 착용 방법을 따르는 것이 중요하다. 그러나 공공장소에서 모든 사람의 마스크 착용 상태를 확인하는 것은 인력적 제약이 있다.

이에 대한 해결책으로, 카메라를 통해 사람의 얼굴 이미지만으로 마스크 착용 여부를 자동으로 판별할 수 있는 시스템의 개발이 필요. 이 시스템이 공공장소 입구에 설치되면, 적은 인력으로도 효과적인 검사가 가능해지고, 이는 COVID-19 확산 방지를 위한 중요한 조치 중 하나가 될 것.

따라서 카메라로 촬영된 얼굴 이미지를 통해 이미지내의 사람의 성별, 연령 그리고 마스크 착용 여부를 분류하는 모델을 개발하는 것이 목표

input: 4,500명의 사람들의 이미지 및 train.csv
output: test 이미지에 대한 분류 값 (18개 클래스)
train.csv
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-12/tree/main/asset/data_1.png)
train image input
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-12/tree/main/asset/data_2.png)
- - -
### 결과
val graph
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-12/tree/main/asset/graph.png)

결 과
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-12/tree/main/asset/result.png)
- - -
### Details

(업데이트 예정)
