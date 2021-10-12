import pandas as pd
import streamlit as st


@st.cache
def load_reports_data(nrows):
    data = pd.read_csv('./kaggle_x-ray/indiana_reports.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data


@st.cache
def load_projections_data(nrows):
    data = pd.read_csv('./kaggle_x-ray/indiana_projections.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data


st.set_page_config(layout='wide')
st.title('AI Assessment System')

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

with st.expander('Standard Metrics'):
    st.subheader('Standard Metrics')
    metrics1, metrics2, metrics3, metrics4 = st.columns(4)
    metrics1.metric(label='Max Error', value='<Value>')
    metrics2.metric(label='Mean Error', value='<Value>')
    metrics3.metric(label='Mean Squared Error', value='<Value>')
    metrics4.metric(label='Root Mean Squared Error', value='<Value>')

projections_data = load_projections_data(100)
reports_data = load_reports_data(100)

with st.expander('Browse Data'):
    st.subheader('Browse Data')
    uid = st.selectbox('Select uid:', projections_data['uid'])
    img_url = projections_data['filename'][uid]
    left_column, right_column = st.columns(2)
    projections_table = left_column.dataframe(projections_data)
    if left_column.checkbox('Show metadata'):
        st.table(reports_data.loc[reports_data['uid'] == uid])
    if left_column.button('Show more'):
        projections_data = load_projections_data(200)
        projections_table.add_rows(projections_data)
    right_column.image(f'./kaggle_x-ray/images/images_normalized/{img_url}', caption=f'{img_url}')

with st.expander('Browse Metadata'):
    st.subheader('Browse Metadata')
    st.write(reports_data)
