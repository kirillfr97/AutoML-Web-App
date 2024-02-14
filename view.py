import streamlit as st
from streamlit_pandas_profiling import st_profile_report


def display_sidebar():
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('AutoStreamML App')
    return st.radio('Navigation', ('Upload', 'Profiling', 'Modeling'))


def display_data_upload():
    st.title('Upload Your Data for Modelling')
    return st.file_uploader('Upload Your Dataset', type='csv')


def display_data_frame(df):
    st.dataframe(df)


def display_profiling_report(report):
    st_profile_report(report)


def display_error(message):
    st.error(message)
