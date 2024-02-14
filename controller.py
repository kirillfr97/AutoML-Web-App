import os
import pandas as pd
from pycaret.classification import save_model

from view import *
from model import load_data, save_data, run_data_analysis, train_model


if __name__ == '__main__':
    # Set up the sidebar UI and load data
    with st.sidebar:
        choice = display_sidebar()
        df = load_data('data/sourcedata.csv')

        # Provide option to download the best model if it exists
        if st.button('Download'):
            if os.path.exists('best_model.pkl'):
                with open('best_model.pkl', 'rb') as f:
                    st.download_button('Download the Model', f, 'automl_best_model.pkl')
        st.info('This application allows you to build an automated ML pipeline using Streamlit, Pandas, and PyCaret.')

    # Perform actions based on user choice
    if choice == 'Upload':
        # Upload data for modeling
        if df is not None:
            st.info('Source data was loaded and is ready for prediction.')
        file = display_data_upload()
        if file is not None:
            df = pd.read_csv(file)
            save_data(df, 'data/sourcedata.csv')
            display_data_frame(df)

    elif choice == 'Profiling':
        # Perform automated data analysis
        st.title('Automated Data Analysis')
        if st.button('Run Data Analysis'):
            if df is not None:
                report = run_data_analysis(df)
                display_profiling_report(report)
            else:
                display_error('Please upload your dataset!')

    elif choice == 'Modeling':
        # Perform machine learning tasks
        st.title('Machine Learning...')
        if df is not None:
            choose_target = st.selectbox('Choose the Target Column', df.columns)
            if st.button('Train Model'):
                setup_df, compare_df, best_model = train_model(df, choose_target)

                # Display ML settings
                st.info('This is the ML settings')
                display_data_frame(setup_df)

                # Display ML Model
                st.info('This is the ML Model')
                display_data_frame(compare_df)

                # Save the best model
                save_model(best_model, 'best_model')
        else:
            display_error('Please upload your dataset!')
