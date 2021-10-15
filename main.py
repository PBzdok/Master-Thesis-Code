import pandas as pd
import streamlit as st
import torchvision
import torchxrayvision as xrv

st.set_page_config(layout='wide')
st.title('AI Assessment System')


@st.cache
def load_label_csv(nrows):
    data = pd.read_csv('./data/kaggle-pneumonia-jpg/stage_2_train_labels.csv', nrows=nrows)
    return data


@st.cache
def load_label_detail_csv(nrows):
    data = pd.read_csv('./data/kaggle-pneumonia-jpg/stage_2_detailed_class_info.csv', nrows=nrows)
    return data


@st.cache
def load_dataset():
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    return xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
                                               transform=transform)


model = xrv.models.DenseNet(weights="densenet121-res224-rsna")  # RSNA Pneumonia Challenge
d_kaggle = load_dataset()

with st.expander('Overview'):
    st.subheader('<Model Name>')
    overview_l, overview_r = st.columns(2)
    overview_l.text('<Import Date>')
    overview_r.text('<Framework>')

with st.expander('Model Description'):
    st.subheader('Model Description')
    st.write(
        'Dolores sunt consequatur laborum. Et rem autem dolores qui assumenda. Sunt illum aut aspernatur maxime quo nostrum illo amet. Reprehenderit ut perspiciatis non alias aut accusantium et. Aspernatur tempore in adipisci pariatur earum et ut.')

with st.expander('Capabilities'):
    st.subheader('Capabilities')
    st.markdown('* Dolores sunt consequatur laborum.')
    st.markdown('* Dolores sunt consequatur laborum.')
    st.markdown('* Dolores sunt consequatur laborum.')
    st.table(d_kaggle.pathologies)

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

    right_column.image(f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{patient_id}.jpg',
                       caption=f'{patient_id}.jpg')
