import skimage
import streamlit as st
import torchvision
import torchxrayvision as xrv


@st.cache
def load_nih_dataset():
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    d_nih = xrv.datasets.NIH_Dataset(imgpath='./data/NIH/images-224',
                                     csvpath='./data/NIH/Data_Entry_2017.csv',
                                     transform=transform)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d_nih)
    return d_nih


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
