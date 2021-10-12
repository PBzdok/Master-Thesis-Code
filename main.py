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

projections_data = load_projections_data(10000)
reports_data = load_reports_data(10000)

with st.expander('Projections data'):
    st.subheader('Projections data')
    uid = st.selectbox('Select uid:', projections_data['uid'])
    img_url = projections_data['filename'][uid]
    st.write(f'Selected URL: {img_url}')
    left_column, right_column = st.columns(2)
    left_column.write(projections_data)
    if left_column.checkbox('Show metadata'):
        left_column.write(reports_data.loc[reports_data['uid'] == uid])
    right_column.image(f'./kaggle_x-ray/images/images_normalized/{img_url}')

with st.expander('Reports data'):
    st.subheader('Reports data')
    st.write(reports_data)
