import datetime

import pandas as pd
import streamlit as st
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

    label_data = load_label_detail_csv(10)
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
