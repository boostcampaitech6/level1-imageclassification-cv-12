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
│   ├── multi_coord_f_trainer.py
│   ├── multi_coord_trainer.py
│   ├── multi_f_trainer.py
│   ├── multi_trainer.py
│   ├── single_trainer.py
│   ├── skf_multi_trainer.py
│   └── skf_single_trainer.py
│
├── dataset.py
├── loss.py
├── model.py
├── train.py
├── preprocess.py
├── inference.py
├── inference_ensemble.py
├── requirements.txt
├── EDA.ipynb
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
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/data_1.png)   
   
train image input   
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/data_2.png)
- - -
### 결과
val graph   
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/graph.png)   
   
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/result.png)
- - -
### Details
##### EDA 및 imbalance 완화
데이터 분포를 확인해보니 전체적으로 데이터가 불균형했다. 특히 남녀에서 age 비율이 불균형이 심했다.   
<p align="center"><img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/imbalance1.png" width="500" height="300"/></p>   
아무런 처리를 하지 않은 데이터셋으로 학습을 진행해보니, 모델이 age를 전혀 구분하지 못하고 있었다.   
<p align="center"><img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/imbalance2.png" width="400" height="200"/></p>   
이를 통해 age class 중 old boundary를 수정하여 어느정도 비율을 맞추도록 했다. 또한 train 과 valid set 의 클래스 비율을 균등하게 나누기 위하여 stratified split 을 사용했고, DataLoader 에서는 불균형을 완화하기 위하여 WeightedRandomSampler 를 이용하여 부족한 class 를 복원추출 하도록 했다.   
<p align="center"><img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/imbalance3.png" width="700" height="400"/></p>   

##### 데이터 전처리
1) 이미지 데이터 가공   
마스크, 나이, 성별을 구분하는데 피사체 뒤의 배경은 학습에 방해될 것이라 판단하여 배경을 지웠다.   
이후 old class에 해당하는 이미지의 얼굴을 Mix up 하는 것을 시도했는데, 얼굴의 크기가 제각각임을 파악했다. 이에 피사체의 위치를 동일하게 맞추는 것이 효과적일 것이라 판단하고 object detection 을 활용하여 얼굴부분만 crop 하고 size 를 256 x 256 으로 맞춰 주었다.
<p align="center"><img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/preprocess1.png" width="500" height="300"/></p>   

3) Data Augmentation   
Augmentation 기법을 팀원 다같이 상의하여 데이터 다양성을 적절히 확보할 수 있는 기법들을 선정했으나, 실제 학습에서 모델에 입력으로 들어가는 데이터를 확인하면서 수정을 거듭했다.
<p align="center"><img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/dataaug.png" width="700" height="400"/></p>   

4) Dataset in Pytorch   
사람 한 명 당 7장의 이미지를 가지고 있는데, 이를 완전히 랜덤하게 섞어서 모델에 넣어주는 것은 모델이 우리가 가진 데이터셋의 분포를 잘 커버하도록 학습 시키는 것이고,   
한 사람이 같은 배치에 들어가는 데이터셋은 같은 사람에 대한 상태를 구분할 수 있고, 본 적이 없는 새로운 사람을 예측하도록 학습 시키는 것이라 판단하여 같은 사람의 이미지가 같은 배치에 들어가도록 Dataset Class 를 사용했다.(MaskSplitByProfileDataset)   
##### 모델
1) multi-classifier   
우리 문제가 마스크/나이/성별을 구분하는 문제이고, 이를 모두 합한 18개의 클래스보다 각 클래스 카테고리 별로 구분하는(3/2/3) 것이 유리할 것이라 판단했다. 따라서 각 카테고리 별로 3개의 classifier head 를 가지도록 모델을 디자인했다.   
모델 실험 후 t-SNE, UMAP 로 시각화해서 보면 mask 와 gender 는 어느 정도 구분이 되고 있으나 age는 구분이 잘 안되는 것을 확인했다. 이에 우리 모델이 age 를 표현하기에 충분히 복잡하지 않다고 판단하여 age에는 좀 더 복잡한 backbone 과 깊은 층의 classifier 를 가지도록 디자인 했다.
<p align="center"><img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/model1.png" width="300" height="300"/></p>   
2) multi-backbone   
3개의 task가 집중해야 되는 부분이 다르고, 같은 feature를 공유하면 각각의 loss에서 계산된 gradient가 서로 상쇄될 수 있다고 판단했다. 이에 따라 여러 개의 backbone을 두어 각각 feature map 을 두거나, 2개의 feature map에 각각 2개, 1개의 head 를 mapping 하는 방식을 비교 실험했다.
<p align="center"><img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/model2.png" width="300" height="300"/></p>   
3) fine tuning   
backbone network 로 ImageNet 으로 pretrained 된 모델들을 쓰면서 ImageNet 데이터셋이 프로젝트 데이터셋과 많이 다르다고 판단했으나, 우리가 가진 데이터셋 수가 적기 때문에 많은 수의 데이터셋으로 분류 task 로 학습된 pretrained model 들의 가중치를 활용하면서
classifier head 보다 1/10 의 학습률로 조금씩 학습시키기로 결정했다.

##### 학습
1) loss   
loss 는 쉬운 문제에는 가중치를 줄이고 어려운 문제에 높은 가중치를 주어 학습에 초점을 맞추는 focal loss를 사용했다. 프로젝트 초반에는 구분하기 어려운 문제인 age에 weight를 주어 weighted sum loss 방식을 사용했다. 그러나 성능이 그리 개선되지 않았고, 하이퍼 파라미터인 loss 가중치를 최적으로 찾기 어려웠다.
이에 따라 loss 별로 gradient를 계산해서 가중치를 갱신하고 다음 loss 는 갱신된 가중치로 새로 계산하여 다시 역전파를 수행하는 방식(coordinate)으로 결정했다.
이후 비교를 해보니 coordinate 방식으로 gradient를 계산하여 가중치를 갱신하는 것이 f1 score 가 0.5876 에서 0.6696 으로 더 좋았다.   
2) optimizer   
Adam을 주로 사용하다가, 프로젝트 후반 오버피팅이 일어나고 있다고 판단하여 weight decay(L2 norm-alization 과 분리된 weight decay) 가 들어간 AdamW 를 사용했다. 이는 EfficientViT 논문에서 사용한 optimizer 이기도 했다.   
3) scheduler   
초기 StepLR 쓰면서 오버피팅이 발생하는 경향이 보여, 안 좋은 local minima 에 빠지고 판단했다. 이에 따라 프로젝트 후반에 step size를 크고 작게 주기별로 주어 오버피팅을 방지하고 안정된 flat minima에 도달할 수 있도록 하는 consineAnnealingLR 을 사용하여 추가 실험을 진행했다.
아쉽게도 시간이 부족하여 결과를 제출하진 못했다.
   
<p align="center"><img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-12/blob/main/asset/wandb.png" width="700" height="400"/></p>   

##### 제출
1) Ensemble   
마지막 제출 전 성능을 끌어올리고, 성능이 더 좋은 모델에 더 큰 가중치를 두어서 더 의견이 반영되도록 weight를 f1 score로 하여 weighted voting 방식의 앙상블을 사용했다.   
- - -
### 팀 & 개인 Wrap Up Report   
📝 [CV_12 Wrap Up Report](https://omiino.notion.site/Image-Classification-Wrap-up-Report-05d3b45d758e451794aa8b0acf23af21?pvs=4, "notion link") 📝
