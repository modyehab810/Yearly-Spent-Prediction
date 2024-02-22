# Importing ToolKits
import re
import relations
import prediction

from time import sleep
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.metrics import mean_absolute_error

import streamlit as st
from streamlit.components.v1 import html
from streamlit_option_menu import option_menu
import warnings

pd.set_option('future.no_silent_downcasting', True)
pd.options.mode.copy_on_write = "warn"


def run():
    st.set_page_config(
        page_title="Yearly Spent Prediction",
        page_icon="💰",
        layout="wide"
    )

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Function To Load Our Dataset
    @st.cache_data
    def load_data(the_file_path):
        df = pd.read_csv(the_file_path)
        df.columns = df.columns.str.replace(" ",  "_").str.replace(".", "")
        df.rename(columns={
            "Time_on_App": "App_Usage",
            "Time_on_Website": "Website_Usage",
            "Length_of_Membership": "Membership_Length",
            "Yearly_Amount_Spent": "Yearly_Spent"
        }, inplace=True)
        return df

    # Function To Load Our Dataset
    @st.cache_data
    def load_linear_regression_model(model_path):
        return pd.read_pickle(model_path)

    df = load_data("Ecommerce_Customers.csv")
    model = load_linear_regression_model(
        "linear_regression_yearly_spent_predictor_v1.pkl")

    # Function To Valid Input Data
    @st.cache_data
    def is_valid_data(d):
        letters = list("qwertyuiopasdfghjklzxcvbnm@!#$%^&*-+~")
        return len(d) >= 2 and not any([i in letters for i in list(d)])

    @st.cache_data
    def validate_test_file(test_file_columns):
        col = "\n".join(test_file_columns).lower()
        pattern = re.compile(
            "\w*\W*session\W*\w*\W*app\W*\w*\W*web\W*\w*\W*membership\W*\w*")

        matches = pattern.findall(col)
        return len("\n".join(matches).split("\n")) == 4

    st.markdown(
        """
    <style>
         .main {
            text-align: center; 
         }
         .st-emotion-cache-16txtl3 h1 {
         font: bold 29px arial;
         text-align: center;
         margin-bottom: 15px
            
         }
         div[data-testid=stSidebarContent] {
         background-color: #111;
         border-right: 4px solid #222;
         padding: 8px!important
         
         }

         div.block-containers{
            padding-top: 0.5rem
         }

         .st-emotion-cache-z5fcl4{
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1.1rem;
            padding-right: 2.2rem;
            overflow-x: hidden;
         }

         .st-emotion-cache-16txtl3{
            padding: 2.7rem 0.6rem;
         }

         .plot-container.plotly{
            border: 1px solid #333;
            border-radius: 6px;
         }

         div.st-emotion-cache-1r6slb0 span.st-emotion-cache-10trblm{
            font: bold 24px tahoma;
         }
         div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }

        div[data-baseweb=select]>div{
            cursor: pointer;
            background-color: #111;
            border: 2px solid #0079FF;
        }

        div[data-baseweb=base-input]{
            background-color: #111;
            border: 4px solid #444;
            border-radius: 5px;
            padding: 5px;
        }

        div[data-testid=stFormSubmitButton]> button{
            width: 40%;
            background-color: #111;
            border: 2px solid #0079FF;
            padding: 18px;
            border-radius: 30px;
            opacity: 0.76;
        }
        div[data-testid=stFormSubmitButton]  p{
            font-weight: bold;
            font-size : 20px
        }

        div[data-testid=stFormSubmitButton]> button:hover{
            opacity: 1;
            border: 2px solid #0079FF;
            color: #fff
        }


    </style>
    """,
        unsafe_allow_html=True
    )

    side_bar_options_style = {
        "container": {"padding": "0!important", "background-color": 'transparent'},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "15px"},
        "nav-link-selected": {"background-color": "#0079FF", "font-size": "15px"},
    }

    sub_options_style = {
        "container": {"padding": "3!important", "background-color": '#101010', "border": "2px solid #0079FF"},
        "nav-link": {"color": "white", "padding": "12px", "font-size": "18px", "text-align": "center", "margin": "0px", },
        "nav-link-selected": {"background-color": "#0079FF"},

    }
    header = st.container()
    content = st.container()

    with st.sidebar:
        st.title("SPENT MONEY :blue[PREDICTION]")
        st.image("imgs/money.png", caption="", width=60)
        page = option_menu(
            menu_title=None,
            options=['Home', 'Relations & Correlarions', 'Prediction'],
            icons=['diagram-3-fill', 'bar-chart-line-fill', "cpu"],
            menu_icon="cast",
            default_index=0,
            styles=side_bar_options_style
        )
        st.write("***")

        data_file = st.file_uploader("Upload Your Dataset (CSV)📂", type="csv")

        if data_file is not None:
            if data_file.name.split(".")[-1].lower() != "csv":
                st.error("Please, Upload CSV FILE ONLY")
            else:
                df = pd.read_csv(data_file)

        # Home Page
        if page == "Home":

            with header:
                st.header('Customer Expected Annual Spend 📈💰')

            with content:
                st.dataframe(df.sample(frac=0.25, random_state=35).reset_index(drop=True),
                             use_container_width=True)

                st.write("***")

                st.subheader("Data Summary Overview 🧐")

                len_numerical_data = df.select_dtypes(
                    include="number").shape[1]
                len_string_data = df.select_dtypes(include="object").shape[1]

                if len_numerical_data > 0:
                    st.subheader("Numerical Data [123]")

                    data_stats = df.describe().T
                    st.table(data_stats)

                if len_string_data > 0:
                    st.subheader("String Data [𝓗]")

                    data_stats = df.select_dtypes(
                        include="object").describe().T
                    st.table(data_stats)

        # Relations & Correlations
        if page == "Relations & Correlarions":

            with header:
                st.header("Correlations Between Data 📉🚀")

            with content:
                st.plotly_chart(relations.create_heat_map(df),
                                use_container_width=True)

                st.plotly_chart(relations.create_scatter_matrix(
                    df), use_container_width=True)

                st.write("***")
                col1, col2 = st.columns(2)
                with col1:
                    first_feature = st.selectbox(
                        "First Feature", options=(df.select_dtypes(
                            include="number").columns.tolist()), index=0).strip()

                temp_columns = df.select_dtypes(
                    include="number").columns.to_list().copy()

                temp_columns.remove(first_feature)

                with col2:
                    second_feature = st.selectbox(
                        "Second Feature", options=(temp_columns), index=0).strip()

                st.plotly_chart(relations.create_relation_scatter(
                    df, first_feature, second_feature), use_container_width=True)

        if page == "Prediction":
            with header:
                st.header("Prediction Model 💰🛍️")
                prediction_option = option_menu(menu_title=None, options=["One Value", 'From File'],
                                                icons=[" "]*4, menu_icon="cast", default_index=0,
                                                orientation="horizontal", styles=sub_options_style)

            with content:
                if prediction_option == "One Value":
                    with st.form("Predict_value"):
                        c1, c2 = st.columns(2)
                        with c1:
                            session_length = st.number_input(label="Average Session Length (Minute)",
                                                             min_value=5.0, max_value=50.0, value=30.0,
                                                             )
                            web_usage_length = st.number_input(label="Time of Website (Minute)",
                                                               min_value=10.0,  value=30.0)
                        with c2:
                            app_usage_length = st.number_input(label="Time of APP (Minute)",
                                                               min_value=8.0, max_value=30.0, value=15.0)
                            membership_length = st.number_input(label="Membership Length (Months)",
                                                                min_value=1.0, value=2.0)

                        st.write("")  # Space

                        predict_button = st.form_submit_button(
                            label='Predict', use_container_width=True)

                        st.write("***")  # Space

                        if predict_button:

                            with st.spinner(text='Predict The Value..'):
                                new_data = [
                                    session_length, app_usage_length, web_usage_length, membership_length]
                                predicted_value = model.predict([new_data])
                                sleep(1.2)

                                predicted_col, score_col = st.columns(2)

                                with predicted_col:
                                    st.image("imgs/money-bag.png",
                                             caption="", width=70)

                                    st.subheader(
                                        "Expected To Spent")
                                    st.subheader(
                                        f"${np.round(predicted_value[0], 2)}")

                                with score_col:
                                    st.image("imgs/star.png",
                                             caption="", width=70)
                                    st.subheader("Model Accuracy")
                                    st.subheader(f"{np.round(98.27, 2)}%")

                if prediction_option == "From File":
                    st.info("Please upload your file with the following columns' names in the same order\n\
                            [Avg_Session_Length, App_Usage, Website_Usage, Membership_Length]", icon="ℹ️")

                    test_file = st.file_uploader(
                        "Upload Your Test File 📂", type="csv")

                    if test_file is not None:
                        extention = test_file.name.split(".")[-1]
                        if extention.lower() != "csv":
                            st.error("Please, Upload CSV FILE ONLY")

                        else:
                            X_test = pd.read_csv(test_file)
                            X_test.columns = X_test.columns.str.replace(
                                " ",  "_").str.replace(".", "")

                            X_test.dropna(inplace=True)

                            if validate_test_file(X_test.columns.to_list()):
                                all_predicted_values = model.predict(
                                    X_test)
                                final_complete_file = pd.concat([X_test, pd.DataFrame(all_predicted_values,
                                                                                      columns=["Predicted_Yearly_Spent"])], axis=1)
                                st.write("")
                                st.dataframe(final_complete_file,
                                             use_container_width=True)
                            else:
                                st.warning(
                                    "Please, Check That Your Test File Has The Mention Columns in The Same Order", icon="⚠️")

                    with st.form("comaprison_form"):

                        if st.form_submit_button("Compare Predicted With Actual Values"):
                            st.info(
                                "Be Sure Your Actual Values File HAS ONLY ONE COLUMN (Yearly_Spent)", icon="ℹ️")

                            actual_file = st.file_uploader(
                                "Upload Your Actual Data File 📂", type="csv")

                            if actual_file is not None and test_file is not None:
                                y_test = pd.read_csv(actual_file)
                                if y_test.shape[1] == 1:

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        test_score = np.round(
                                            model.score(X_test, y_test) * 100, 2)
                                        prediction.creat_matrix_score_cards("imgs/star.png",
                                                                            "Prediction Accuracy",
                                                                            test_score,
                                                                            True
                                                                            )

                                    with col2:
                                        mae = mean_absolute_error(
                                            y_test, all_predicted_values)
                                        prediction.creat_matrix_score_cards("imgs/sort.png",
                                                                            "Error Ratio",
                                                                            np.round(
                                                                                mae, 2),
                                                                            False)

                                    predicted_df = prediction.create_comparison_df(
                                        y_test, all_predicted_values)
                                    st.dataframe(
                                        predicted_df, use_container_width=True, height=300)

                                    st.plotly_chart(prediction.create_residules_scatter(predicted_df),
                                                    use_container_width=True)

                                else:
                                    st.warning(
                                        "Please, Check That Your Test File Has The One Column.", icon="⚠️")

                            else:
                                st.warning(
                                    "Please, Check That You Upload The Test File & Actual Value", icon="⚠️")


run()
