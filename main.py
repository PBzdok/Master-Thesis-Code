import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchxrayvision as xrv

from data import load_dataset, load_class_csv, image_preprocessing

st.set_page_config(layout='wide')

model_specifier = 'densenet121-res224-rsna'
model = xrv.models.DenseNet(weights=model_specifier)  # RSNA Pneumonia Challenge
d_kaggle = load_dataset()
df_class = load_class_csv()

st.title('AI Assessment System')

# -----------------------------------------------------------------------------------------------------------

with st.expander('Overview'):
    st.subheader(f'{model_specifier}'.upper())
    overview_l, overview_r = st.columns(2)
    overview_l.text(f'{datetime.date.today()}')
    overview_r.text('Pytorch')

# -----------------------------------------------------------------------------------------------------------

with st.expander('Model Description'):
    st.subheader('Model Description')
    st.write(
        'Dolores sunt consequatur laborum. Et rem autem dolores qui assumenda. Sunt illum aut aspernatur maxime quo nostrum illo amet. Reprehenderit ut perspiciatis non alias aut accusantium et. Aspernatur tempore in adipisci pariatur earum et ut.')

# -----------------------------------------------------------------------------------------------------------

with st.expander('Capabilities'):
    st.subheader('Capabilities')
    st.write('The model is able to predict following pathologies from image data:')
    for p in d_kaggle.pathologies:
        st.markdown(f'* {p}')

# -----------------------------------------------------------------------------------------------------------

with st.expander('Standard Metrics'):
    st.subheader('Standard Metrics')
    metrics1, metrics2, metrics3, metrics4 = st.columns(4)
    metrics1.metric(label='Max Error', value='<Value>')
    metrics2.metric(label='Mean Error', value='<Value>')
    metrics3.metric(label='Mean Squared Error', value='<Value>')
    metrics4.metric(label='Root Mean Squared Error', value='<Value>')


# -----------------------------------------------------------------------------------------------------------

def set_browse_indices(high):
    st.session_state['indices'] = np.random.randint(low=0, high=high, size=10)


with st.expander('Browse Data'):
    st.subheader('Browse Data')

    labels = df_class['class'].unique()
    labels_columns = st.columns(len(labels))
    options = []
    for _, (label, column) in enumerate(zip(labels, labels_columns)):
        if column.checkbox(label, value=True):
            options.append(label)

    if 'indices' not in st.session_state:
        set_browse_indices(len(d_kaggle.csv.index))
    index_list = st.session_state['indices']
    df_class_options = df_class.loc[df_class.index[index_list]][df_class['class'].isin(options)]

    idx = st.selectbox('Select row:', df_class_options.index)
    patient_id = df_class_options['patientId'][idx]

    left_column, right_column = st.columns(2)
    left_column.table(df_class_options['class'])
    if left_column.checkbox('Show metadata'):
        st.table(d_kaggle.csv.loc[d_kaggle.csv['patientId'] == patient_id][
                     ['BodyPartExamined', 'PatientAge', 'PatientOrientation', 'PatientSex', 'PixelSpacing',
                      'SamplesPerPixel', 'ViewPosition']])
        # st.dataframe(d_kaggle.csv)

    if left_column.button('Show more images'):
        set_browse_indices(len(d_kaggle.csv.index))

    right_column.image(f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{patient_id}.jpg',
                       caption=f'{patient_id}.jpg')


# -----------------------------------------------------------------------------------------------------------

def set_experiment_index():
    st.session_state['index'] = np.random.randint(low=0, high=len(d_kaggle.csv.index), size=1)


with st.expander('Experiment'):
    st.subheader('Experiment')

    if 'index' not in st.session_state:
        set_experiment_index()
    index = st.session_state['index']

    experiment_left_column, experiment_right_column = st.columns(2)
    experiment_sample = df_class.loc[df_class.index[index]]
    experiment_sample_id = experiment_sample['patientId'][index[0]]

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
        set_experiment_index()
