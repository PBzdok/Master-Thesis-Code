import numpy as np
import pandas as pd
import skimage
import sklearn.metrics
import streamlit as st
import torch
import torchvision
import torchxrayvision as xrv

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])


@st.cache
def load_nih_dataset():
    d_nih = xrv.datasets.NIH_Dataset(imgpath='./data/NIH/images-224',
                                     csvpath='./data/NIH/Data_Entry_2017.csv',
                                     transform=transform)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d_nih)
    return d_nih


@st.cache
def load_rsna_dataset():
    d_kag = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath='./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg',
                                                views=["PA", "AP"],
                                                unique_patients=True,
                                                transform=transform)
    return d_kag


@st.cache
def load_detailed_rsna_class_info():
    return pd.read_csv('./data/kaggle-pneumonia-jpg/stage_2_detailed_class_info.csv')


@st.cache
def load_cluster_metadata():
    return pd.read_csv('./data/kaggle-pneumonia-jpg/metadata_with_clusters.csv')


@st.cache
def calculate_rsna_metrics(model, dataset):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i in np.random.randint(0, len(dataset), 1000):
            sample = dataset[i]
            y_true.append(sample["lab"][0])
            out = model(torch.from_numpy(sample["img"]).unsqueeze(0)).cpu()
            # out = torch.sigmoid(out)
            out = (out > 0.5).float()
            out = out.detach().numpy()[0]
            out = out[8]
            print(out)
            y_pred.append(out)

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1


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
