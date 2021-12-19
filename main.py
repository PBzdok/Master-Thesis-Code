import datetime

import numpy as np
import streamlit as st
import torchxrayvision as xrv

from data import load_rsna_dataset, load_detailed_rsna_class_info, load_cluster_metadata, calculate_rsna_metrics

st.set_page_config(layout='wide')

model_specifier = 'densenet121-res224-rsna'
model = xrv.models.DenseNet(weights=model_specifier)
d_rsna = load_rsna_dataset()

detailed_class_info = load_detailed_rsna_class_info()
classes = detailed_class_info['class'].unique()

cluster_metadata = load_cluster_metadata()

dataset = d_rsna.csv.merge(detailed_class_info[['patientId', 'class']], on='patientId')
dataset = dataset.merge(cluster_metadata[['anomaly_score', 'cluster', 'patientId']], on='patientId')

df_predictions, metrics = calculate_rsna_metrics(model, d_rsna)

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
        'The "DENSENET121-RES224-RSNA" AI model is a variation of a Densely Connected Convolutional Network architecture. '
        'This architecture has been shown to generate the best predictive performance for X-Ray classification tasks. '
        'The Model is pre-trained with large amounts of publicly available X-Ray data and was supplied by the Machine Learning and Medicine lab (mlmed.org). '
        'The model expects an input of 224x224 pixels (grayscale) and outputs the probability of 18 different pathologies, although this version is specialised on pneumonia and predicts only that.'
    )

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
    metrics1.metric(label='Accuracy', value=str(metrics['accuracy'].round(2)))
    metrics2.metric(label='Precision', value=str(metrics['precision'].round(2)))
    metrics3.metric(label='Sensitivity', value=str(metrics['recall'].round(2)))
    metrics4.metric(label='F1', value=str(metrics['f1'].round(2)))


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

    df_samples = dataset.loc[index_list]
    df_samples = df_samples[df_samples['class'].isin(options)]

    selected_idx = st.selectbox('Select row:', df_samples.index.values)
    patient_id = df_samples['patientid'][selected_idx]
    image_id = patient_id

    df_samples = df_samples[['class', 'Target']]
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

def set_cluster_index(index, cluster):
    st.session_state.cluster_index = index
    set_cluster_sample_index(cluster.index)


def set_cluster_sample_index(indices):
    st.session_state.cluster_sample_index = np.random.choice(indices, 1)


with st.expander('Data Clusters'):
    st.subheader('Data Clusters')

    cluster_labels = dataset['cluster'].unique()
    cluster_labels = np.sort(cluster_labels)
    n_clusters = len(cluster_labels)

    clusters = dataset.groupby('cluster')

    st.write(f'A dataset analysis has shown that {n_clusters} main clusters of data can be aggregated by metadata.')

    left_cluster_column, mid_cluster_column, right_cluster_column = st.columns([0.5, 2, 3])

    if 'cluster_index' not in st.session_state:
        set_cluster_index(0, clusters.get_group(0))
    cluster_index = st.session_state.cluster_index
    selected_cluster = clusters.get_group(cluster_index)

    for c in cluster_labels:
        left_cluster_column.button(f'Cluster {c}', key=f'Cluster {c}', on_click=set_cluster_index,
                                   args=(c, clusters.get_group(c),))

    mean_anomaly_score = selected_cluster['anomaly_score'].mean()
    mean_age = selected_cluster['PatientAge'].mean()
    max_age = selected_cluster['PatientAge'].max()
    min_age = selected_cluster['PatientAge'].min()
    n_men = sum(selected_cluster['PatientSex'] == 'M')
    n_women = sum(selected_cluster['PatientSex'] == 'F')
    n_pathological_cluster = sum(selected_cluster['Target'] == 1)
    n_non_pathological_cluster = sum(selected_cluster['Target'] == 0)

    mid_cluster_column.markdown(
        f'* Cluster {cluster_index} has {len(selected_cluster.index)} entries and a mean anomaly score of {mean_anomaly_score.round(2)}.\n'
        f'* The mean age of this cluster is {mean_age.round(2)}, while the maximum and minimum age are {max_age} and {min_age} respectively.\n'
        f'* This cluster contains {n_men} male instances and {n_women} female instances.\n'
        f'* This cluster contains {n_pathological_cluster} pathological instances and {n_non_pathological_cluster} non pathological instances.\n'
    )

    cluster_sample_index = st.session_state.cluster_sample_index
    cluster_sample_id = selected_cluster['patientid'][cluster_sample_index].values[0]
    image_path = f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{cluster_sample_id}.jpg'
    right_cluster_column.image(image_path)
    right_cluster_column.table(dataset.loc[dataset['patientid'] == cluster_sample_id][
                                   ['PatientAge', 'PatientSex', 'Target', 'anomaly_score']])
    right_cluster_column.button(f'Show another examples from cluster {cluster_index}',
                                on_click=set_cluster_sample_index, args=(selected_cluster.index,))


# -----------------------------------------------------------------------------------------------------------

def set_fp_indices(indices):
    st.session_state.fp_indices = np.random.choice(indices, 10)


def set_fn_indices(indices):
    st.session_state.fn_indices = np.random.choice(indices, 10)


with st.expander('Model Limitations'):
    st.subheader('Model Limitations')

    df_limit_samples = df_predictions
    fp_index_list = []
    fn_index_list = []

    case = st.radio(
        'Choose Edge Case:',
        ['False Positive', 'False Negative']
    )
    if case == 'False Positive':
        df_limit_samples = df_limit_samples[(df_limit_samples['y_true'] == 0) & (df_limit_samples['y_pred'] == 1)]
        if 'fp_indices' not in st.session_state:
            set_fp_indices(df_limit_samples.index.values)
        fp_index_list = st.session_state.fp_indices
        df_limit_samples = df_limit_samples.loc[fp_index_list]
    else:
        df_limit_samples = df_limit_samples[(df_limit_samples['y_true'] == 1) & (df_limit_samples['y_pred'] == 0)]
        if 'fn_indices' not in st.session_state:
            set_fn_indices(df_limit_samples.index.values)
        fn_index_list = st.session_state.fn_indices
        df_limit_samples = df_limit_samples.loc[fn_index_list]

    selected_limit_idx = st.selectbox('Select row:', df_limit_samples.index)
    patient_limit_id = df_limit_samples.drop_duplicates()['patientid'][selected_limit_idx]
    image_limit_id = patient_limit_id

    df_limit_samples = df_limit_samples[['y_pred', 'y_true']].astype(int)

    df_limit_samples = df_limit_samples.rename(columns={'y_pred': 'Prediction', 'y_true': 'Evidence of Pneumonia'})

    left_limit_column, right_limit_column = st.columns(2)
    right_limit_column.table(df_limit_samples)
    if right_limit_column.checkbox('Show metadata', key='limits'):
        st.table(dataset.loc[dataset['patientid'] == patient_limit_id][
                     ['PatientAge', 'PatientSex', 'ViewPosition', 'BodyPartExamined', 'ConversionType']])

    image_path = f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{image_limit_id}.jpg'
    left_limit_column.image(image_path, caption=f'{image_limit_id}')

    if left_limit_column.checkbox('Show explanation based on which image areas the AI decided'):
        st.image(f'./data/kaggle-pneumonia-jpg/occlusion/{image_limit_id}.jpg',
                 caption="Bright / Yellow areas show areas of high importance!")


# -----------------------------------------------------------------------------------------------------------

def set_experiment_index(high):
    st.session_state.index = np.random.randint(low=0, high=high, size=1)


def show_occlusion(id):
    st.image(f'./data/kaggle-pneumonia-jpg/occlusion/{id}.jpg',
             caption="Bright / Yellow areas show areas of high importance!")


with st.expander('Experiment'):
    st.subheader('Experiment')

    if 'index' not in st.session_state:
        set_experiment_index(len(df_predictions.index))
    rnd_idx = st.session_state.index

    sample = df_predictions.iloc[rnd_idx[0]]
    sample_id = sample['patientid']
    sample_pred = sample['y_pred']

    experiment_sample_path = f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{sample_id}.jpg'

    experiment_left_column, experiment_right_column = st.columns(2)
    experiment_left_column.image(experiment_sample_path, caption=f'{sample_id}.jpg')

    prediction = experiment_right_column.radio('Do you think the image is pathological?', ('Yes', 'No'))

    if experiment_right_column.button('Show Result'):
        if (prediction == 'Yes' and sample_pred == 1.0) or (prediction == 'No' and sample_pred == 0.0):
            experiment_right_column.success('You and the AI have the same opinion!')
        else:
            experiment_right_column.error('You and the AI have different opinions!')

    experiment_right_column.button('Try Again', on_click=set_experiment_index, args=(len(df_predictions.index),))

    if experiment_right_column.checkbox('Show metadata', key='experiment'):
        experiment_right_column.table(dataset.loc[dataset['patientId'] == sample_id][
                                          ['PatientAge', 'PatientSex', 'ViewPosition', 'BodyPartExamined']])
    if experiment_right_column.checkbox('Show explanation based on which image areas the AI decided',
                                        key='experiment'):
        st.image(f'./data/kaggle-pneumonia-jpg/occlusion/{sample_id}.jpg',
                 caption="Bright / Yellow areas show areas of high importance!")
