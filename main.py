import datetime

import numpy as np
import streamlit as st
import torchxrayvision as xrv
from PIL import Image

from data import load_nih_dataset

st.set_page_config(layout='wide')

model_specifier = 'densenet121-res224-all'
model = xrv.models.DenseNet(weights='all')
d_nih = load_nih_dataset()

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
    for p in model.pathologies:
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
    st.session_state.indices = np.random.randint(low=0, high=high, size=10)


with st.expander('Browse Data'):
    st.subheader('Browse Data')

    labels = d_nih.pathologies

    rows = list()
    chunk_size = 3

    for i in range(0, len(labels), chunk_size):
        rows.append(labels[i:i + chunk_size])

    for row in rows:
        labels_columns = st.columns(len(row))
        options = []
        for _, (label, column) in enumerate(zip(row, labels_columns)):
            if column.checkbox(label, value=True, key=label):
                options.append(label)

    if 'indices' not in st.session_state:
        set_browse_indices(len(d_nih.csv.index))
    index_list = st.session_state.indices

    df_samples = d_nih.csv.loc[d_nih.csv.index[index_list]]

    idx = st.selectbox('Select row:', df_samples.index)
    patient_id = df_samples['patientid'][idx]
    image_id = df_samples['Image Index'][idx]

    left_column, right_column = st.columns(2)
    left_column.table(df_samples['Finding Labels'])
    if left_column.checkbox('Show metadata'):
        st.table(d_nih.csv.loc[d_nih.csv['patientid'] == patient_id][
                     ['Follow-up #', 'Patient Age', 'Patient Gender', 'View Position']])

    left_column.button('Show more images', on_click=set_browse_indices, args=(len(d_nih.csv.index),))

    image_path = f'./data/NIH/images-224/{image_id}'
    image = Image.open(image_path)
    resized_image = image.resize((672, 672), Image.BILINEAR)
    right_column.image(resized_image, caption=f'{image_id}')

# -----------------------------------------------------------------------------------------------------------

# def set_experiment_index():
#     st.session_state['index'] = np.random.randint(low=0, high=len(d_nih.csv.index), size=1)
#
#
# with st.expander('Experiment'):
#     st.subheader('Experiment')
#
#     if 'index' not in st.session_state:
#         set_experiment_index()
#     index = st.session_state['index']
#
#     experiment_left_column, experiment_right_column = st.columns(2)
#     experiment_sample = df_details.loc[df_details.index[index]]
#     experiment_sample_id = experiment_sample['patientId'][index[0]]
#
#     experiment_left_column.image(f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{experiment_sample_id}.jpg',
#                                  caption=f'{experiment_sample_id}.jpg')
#     with torch.no_grad():
#         out = model(torch.from_numpy(
#             image_preprocessing(
#                 f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{experiment_sample_id}.jpg'))
#                     .unsqueeze(0)) \
#             .cpu()
#         result = dict(zip(model.pathologies,
#                           out[0].detach().numpy()))
#         result.pop("")
#         df_result = pd.DataFrame(list(result.items()), columns=['Pathology', 'Prediction'])
#
#         prediction = experiment_right_column.radio('Do you think the image is pathological?', ('Yes', 'No'))
#         if experiment_right_column.button('Show Result'):
#             experiment_right_column.table(df_result)
#
#     if st.button("Reload"):
#         set_experiment_index()
#
# with st.expander('Test'):
#     st.subheader('Test')
#     sample = d_nih[0]
#     plt.imshow(sample["img"][0], cmap="Greys_r")
#     dict(zip(d_nih.pathologies, sample["lab"]))
