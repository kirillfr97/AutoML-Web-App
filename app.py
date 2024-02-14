import os
import pandas as pd
import streamlit as st

# Import profiling capability
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML
from pycaret.classification import setup, compare_models, pull, save_model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with st.sidebar:
        st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
        st.title('AutoStreamML App')
        choice = st.radio('Navigation', ('Upload', 'Profiling', 'Modeling'))
        if st.button('Download'):
            if os.path.exists('best_model.pkl'):
                with open('best_model.pkl', 'rb') as f:
                    st.download_button('Download the Model', f, 'automl_best_model.pkl')
        st.info('This application allows you to build an automated ML pipeline using Streamlit, Pandas, and PyCaret.')

    df = None
    if os.path.exists('data/sourcedata.csv'):
        df = pd.read_csv('data/sourcedata.csv', index_col=None)

    if choice == 'Upload':
        st.title('Upload Your Data for Modelling')
        file = st.file_uploader('Upload Your Dataset', type='csv')
        if file is not None:
            df = pd.read_csv(file, index_col=None)
            df.to_csv('sourcedata.csv')
            st.dataframe(df)

    if choice == 'Profiling':
        st.title('Automated Data Analysis')
        if st.button('Run Data Analysis'):
            if df is not None:
                report = ProfileReport(df, title='Profiling Report')
                st_profile_report(report)
            else:
                st.error('Please upload your dataset!')

    if choice == 'Modeling':
        st.title('Machine Learning...')
        if df is not None:
            choose_target = st.selectbox('Choose the Target Column', df.columns)
            if st.button('Train Model'):
                setup(df, target=choose_target)

                setup_df = pull()
                st.info('This is the ML settings')
                st.dataframe(setup_df)

                best_model = compare_models()
                compere_df = pull()
                st.info('This is the ML Model')
                st.dataframe(compere_df)

                save_model(best_model, 'best_model')
        else:
            st.error('Please upload your dataset!')
