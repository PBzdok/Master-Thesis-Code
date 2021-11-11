import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchxrayvision as xrv

from data import load_rsna_dataset, load_detailed_rsna_class_info, load_cluster_metadata, calculate_rsna_metrics

st.set_page_config(layout='wide')

model_specifier = 'densenet121-res224-rsna'
model = xrv.models.DenseNet(weights='densenet121-res224-rsna')
d_rsna = load_rsna_dataset()

detailed_class_info = load_detailed_rsna_class_info()
classes = detailed_class_info['class'].unique()

cluster_metadata = load_cluster_metadata()

dataset = d_rsna.csv.merge(detailed_class_info[['patientId', 'class']], on='patientId')
dataset = dataset.merge(cluster_metadata[['anomaly_score', 'cluster', 'patientId']], on='patientId')

# metrics = calculate_rsna_metrics(model, d_rsna)

st.title('AI Assessment System')

# -----------------------------------------------------------------------------------------------------------

with st.expander('Overview'):
    st.subheader(f'{model_specifier}'.upper())
    overview_l, overview_r = st.columns(2)
    overview_l.text(f'Import Date: {datetime.date.today()}')
    overview_r.text('Framework: Pytorch')

# -----------------------------------------------------------------------------------------------------------

with st.expander('Model Description'):
    st.subheader('Model Description')
    st.write(
        'Dolores sunt consequatur laborum. Et rem autem dolores qui assumenda. Sunt illum aut aspernatur maxime quo nostrum illo amet. Reprehenderit ut perspiciatis non alias aut accusantium et. Aspernatur tempore in adipisci pariatur earum et ut.')

# -----------------------------------------------------------------------------------------------------------

with st.expander('Capabilities'):
    st.subheader('Capabilities')
    st.write('The model is able to predict following pathologies from x-ray images:')
    for p in d_rsna.pathologies:
        st.markdown(f'* {p}')

# -----------------------------------------------------------------------------------------------------------

with st.expander('Standard Metrics'):
    st.subheader('Standard Metrics')
    metrics1, metrics2, metrics3, metrics4 = st.columns(4)
    metrics1.metric(label='Accuracy', value='0.62')
    metrics2.metric(label='Precision', value='0.32')
    metrics3.metric(label='Sensitivity', value='0.94')
    metrics4.metric(label='F1', value='0.47')


# -----------------------------------------------------------------------------------------------------------

def set_browse_indices(high):
    st.session_state.indices = np.random.randint(low=0, high=high, size=10)


with st.expander('Browse Data'):
    st.subheader('Browse Data')
    n_pathological = sum(dataset['Target'] == 1)
    n_non_pathological = sum(dataset['Target'] == 0)
    st.write(
        f'This dataset contains {n_pathological} pathological images and {n_non_pathological} non pathological images')

    rows = list()
    chunk_size = 3
    options = []

    for i in range(0, len(classes), chunk_size):
        rows.append(classes[i:i + chunk_size])

    for row in rows:
        labels_columns = st.columns(len(row))
        for _, (label, column) in enumerate(zip(row, labels_columns)):
            if column.checkbox(label, value=True, key=label):
                options.append(label)

    if 'indices' not in st.session_state:
        set_browse_indices(len(dataset.index))
    index_list = st.session_state.indices

    df_samples = dataset.loc[dataset.index[index_list]]

    selected_idx = st.selectbox('Select row:', df_samples.index)
    patient_id = df_samples['patientid'][selected_idx]
    image_id = patient_id

    df_samples = df_samples[df_samples['class'].isin(options)][['class', 'Target']]
    df_samples = df_samples.rename(columns={'class': 'Details', 'Target': 'Evidence of Pneumonia'})

    left_column, right_column = st.columns(2)
    left_column.table(df_samples)
    if left_column.checkbox('Show metadata'):
        st.table(dataset.loc[dataset['patientid'] == patient_id][
                     ['PatientAge', 'PatientSex', 'ViewPosition', 'BodyPartExamined', 'ConversionType']])

    left_column.button('Show more images', on_click=set_browse_indices, args=(len(dataset.index),))

    image_path = f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{image_id}.jpg'
    right_column.image(image_path, caption=f'{image_id}')


# -----------------------------------------------------------------------------------------------------------

def set_experiment_index(high):
    st.session_state.index = np.random.randint(low=0, high=high, size=1)


with st.expander('Experiment'):
    st.subheader('Experiment')

    if 'index' not in st.session_state:
        set_experiment_index(len(dataset.index))
    rnd_idx = st.session_state.index

    sample = d_rsna[rnd_idx[0]]
    sample_id = d_rsna.csv['patientId'][sample['idx']]

    experiment_sample_path = f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{sample_id}.jpg'

    experiment_left_column, experiment_right_column = st.columns(2)
    experiment_left_column.image(experiment_sample_path, caption=f'{sample_id}.jpg')
    with torch.no_grad():
        out = model(torch.from_numpy(sample['img']).unsqueeze(0)).cpu()
        out = torch.sigmoid(out)

        result = dict(zip(model.pathologies, out[0].detach().numpy()))
        result.pop("")
        df_result = pd.DataFrame(list(result.items()), columns=['Pathology', 'Prediction'])

        prediction = experiment_right_column.radio('Do you think the image is pathological?', ('Yes', 'No'))
        # if experiment_right_column.button('Show Result'):
        experiment_right_column.table(df_result)
        if experiment_right_column.checkbox('Show metadata', key='experiment'):
            experiment_right_column.table(dataset.loc[dataset['patientId'] == sample_id][
                                              ['PatientAge', 'PatientSex', 'ViewPosition', 'BodyPartExamined']])

    st.button('Reload', on_click=set_experiment_index, args=(len(dataset.index),))
