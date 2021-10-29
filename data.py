import pandas as pd
import skimage
import streamlit as st
import torchvision
import torchxrayvision as xrv


@st.cache
def load_train_label_csv():
    data = pd.read_csv('./data/kaggle-pneumonia-jpg/stage_2_train_labels.csv')
    return data


@st.cache
def load_class_csv():
    data = pd.read_csv('./data/kaggle-pneumonia-jpg/stage_2_detailed_class_info.csv')
    return data


@st.cache
def load_dataset():
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    return xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
                                               transform=transform,
                                               unique_patients=True)


def image_preprocessing(img_path):
    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, 255)

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])

    return transform(img)
