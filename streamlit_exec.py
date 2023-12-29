import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from model import *

def load_data(image):
    mean = (0.2409401, 0.19878025, 0.186334)
    std = (0.29069433, 0.25123745, 0.24154018)
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    img = transform(image)
    img = torch.unsqueeze(img, dim=0)
    return img

@st.cache_resource
def load_model(device):
    model = EfficientNetViT_b4(num_classes=18).to(device)
    model_path = "/Users/baekkwanghyun/Desktop/Team_Project/Naver_boostcamp/level1_project/level1-imageclassification-cv-12/model/7088.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model

def get_pred(model, image, device):
    image = image.to(device)
    mask_output, gender_output, age_output = model(image)
    mask_pred = torch.argmax(mask_output, dim=-1)
    gender_pred = torch.argmax(gender_output, dim=-1)
    age_pred = torch.argmax(age_output, dim=-1)
    pred = mask_pred * 6 + gender_pred * 3 + age_pred

    return pred

def decode_multi_class(multi_class_label):
    mask_label = (multi_class_label // 6) % 3
    gender_label = (multi_class_label // 3) % 2
    age_label = multi_class_label % 3
    return mask_label, gender_label, age_label

def meaning_mask_label(mask_label):
    if mask_label == 0:
        return 'You wore your mask correctly! :sunglasses:'
    elif mask_label == 1:
        return 'Please wear a mask to cover your nose and mouth.'
    else:
        return 'Please help prevent COVID-19 by wearing a mask.'
    
def meaning_gender_label(gender_label):
    if gender_label == 0:
        return 'Male'
    else:
        return 'Female'
    
def meaning_age_label(age_label):
    if age_label == 0:
        return 'under 30 years old'
    elif age_label == 1:
        return 'over 30 years old but under 60 years old'
    else:
        return 'over 60 years old'

st.title('Level 1 Image Classification')
st.markdown(':grey[CV_12 백광현, 김시웅, 조형서, 이동형, 박정민]')
st.balloons()

st.subheader('Did you wear your mask well?', divider='rainbow')
st.subheader('_Wearing a mask correctly_ is the :blue[shortcut] to preventing COVID-19.:smile:')

# device 를 자신의 환경에 따라 맞춰주세요. cpu, mps, cuda
device = torch.device("cpu" if torch.backends.mps.is_available() else "mps")
model = load_model(device)
for param in model.parameters():
    param.to(device)
model.eval()

st.markdown('Show me your state :arrow_down:')
picture = st.camera_input("Take a picture!")
if picture is not None:
    img = Image.open(picture).convert('RGB')
    img_tensor = load_data(img)
    with st.spinner("Classifying..."):
        pred = get_pred(model, img_tensor, device)

    mask_label, gender_label, age_label = decode_multi_class(pred)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_area("Mask", meaning_mask_label(mask_label))
    with col2:
        st.text_area("Gender", meaning_gender_label(gender_label))
    with col3:
        st.text_area("Age", meaning_age_label(age_label))
        
    if mask_label != 0:
        st.markdown('''
                    How to use the mask correctly. :mask:
                    
                    1. Cover your nose.
                    2. Completely covers the mouth and chin.
                    3. Tighten so that there are no gaps between the face and the mask.
                    
                    See below! :arrow_down:
                    ''')
        col_1, col_2 = st.columns(2)
        with col_1:
            st.image('https://newsimg.sedaily.com/2020/05/26/1Z2WHN2GNC_1.jpg')
        with col_2:
            st.image('https://www.gangnam.go.kr/upload/editor/2020/08/24/683bc182-5cc5-438f-b6a1-3537076cea2d.jpg')
