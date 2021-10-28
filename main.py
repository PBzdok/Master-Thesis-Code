import datetime

import pandas as pd
import skimage
import streamlit as st
import torch
import torchvision
import torchxrayvision as xrv

st.set_page_config(layout='wide')
st.title('AI Assessment System')


@st.cache
def load_label_csv(nrows, sample=False):
    data = pd.read_csv('./data/kaggle-pneumonia-jpg/stage_2_train_labels.csv', nrows=nrows)
    if sample:
        data = data.sample(nrows)
    return data


@st.cache
def load_label_detail_csv(nrows, sample=False):
    data = pd.read_csv('./data/kaggle-pneumonia-jpg/stage_2_detailed_class_info.csv', nrows=nrows)
    if sample:
        data = data.sample(nrows)
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


model_specifier = 'densenet121-res224-rsna'
model = xrv.models.DenseNet(weights=model_specifier)  # RSNA Pneumonia Challenge
d_kaggle = load_dataset()

with st.expander('Overview'):
    st.subheader(f'{model_specifier}'.upper())
    overview_l, overview_r = st.columns(2)
    overview_l.text(f'{datetime.date.today()}')
    overview_r.text('Pytorch')

with st.expander('Model Description'):
    st.subheader('Model Description')
    st.write(
        'Dolores sunt consequatur laborum. Et rem autem dolores qui assumenda. Sunt illum aut aspernatur maxime quo nostrum illo amet. Reprehenderit ut perspiciatis non alias aut accusantium et. Aspernatur tempore in adipisci pariatur earum et ut.')

with st.expander('Capabilities'):
    st.subheader('Capabilities')
    st.write('The model is able to predict following pathologies from image data:')
    for p in d_kaggle.pathologies:
        st.markdown(f'* {p}')

with st.expander('Standard Metrics'):
    st.subheader('Standard Metrics')
    metrics1, metrics2, metrics3, metrics4 = st.columns(4)
    metrics1.metric(label='Max Error', value='<Value>')
    metrics2.metric(label='Mean Error', value='<Value>')
    metrics3.metric(label='Mean Squared Error', value='<Value>')
    metrics4.metric(label='Root Mean Squared Error', value='<Value>')

with st.expander('Browse Data'):
    st.subheader('Browse Data')

    label_data = load_label_detail_csv(50)
    labels = label_data['class'].unique()
    labels_columns = st.columns(len(labels))

    options = []
    for _, (label, column) in enumerate(zip(labels, labels_columns)):
        if column.checkbox(label, value=True):
            options.append(label)

    label_data = label_data[label_data['class'].isin(options)]

    idx = st.selectbox('Select row:', label_data.index)
    patient_id = label_data['patientId'][idx]

    left_column, right_column = st.columns(2)
    left_column.dataframe(label_data)
    if left_column.checkbox('Show metadata'):
        st.table(d_kaggle.csv.loc[d_kaggle.csv['patientId'] == patient_id][
                     ['BodyPartExamined', 'PatientAge', 'PatientOrientation', 'PatientSex', 'PixelSpacing',
                      'SamplesPerPixel', 'ViewPosition']])
        # st.dataframe(d_kaggle.csv)

    if left_column.button('Show more images'):
        label_data = load_label_detail_csv(10, sample=True)

    right_column.image(f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{patient_id}.jpg',
                       caption=f'{patient_id}.jpg')

with st.expander('Experiment'):
    st.subheader('Experiment')
    experiment_left_column, experiment_right_column = st.columns(2)
    experiment_sample = load_label_detail_csv(1000).sample().reset_index()
    experiment_sample_id = experiment_sample['patientId'][0]

    experiment_left_column.image(f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{experiment_sample_id}.jpg',
                                 caption=f'{experiment_sample_id}.jpg')
    with torch.no_grad():
        out = model(torch.from_numpy(
            image_preprocessing(
                f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{experiment_sample_id}.jpg'))
                    .unsqueeze(0)) \
            .cpu()
        result = dict(zip(model.pathologies,
                          out[0].detach().numpy()))
        result.pop("")
        df_result = pd.DataFrame(list(result.items()), columns=['Pathology', 'Prediction'])

        if experiment_right_column.button('Show Result'):
            experiment_right_column.table(df_result)

        if st.button("Reload"):
            experiment_sample_id = load_label_detail_csv(1000).sample()
