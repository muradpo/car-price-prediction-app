import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Ridge
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Car price prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    with open('model_30_11_ridge.pkl', 'rb') as f:
        saved = pickle.load(f)
    model = saved['model']
    feature_names = saved['feature_names']
    return model, feature_names 

def prepare_features(df, feature_names):
    """Приводим данные к формату обучения модели"""
    df_proc = df.copy()
    # Преобразуем категориальные признаки в строки (как при обучении)
    for col in feature_names:
        if col in df_proc.columns:
            if df_proc[col].dtype in ('object', 'bool'):
                df_proc[col] = df_proc[col].astype(str)
    return df_proc[feature_names]

# Загружаем модель
model, feature_names = load_model()
menu = st.sidebar.radio("Navigation", ["EDA", "Prediction", "Weights"])


if menu == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    df = pd.read_csv("df_train.csv")


    st.subheader("Dataset preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Summary statistics (numeric features)")
    st.write(df.describe())

    st.subheader("Summary statistics (categorical features)")
    st.write(df.describe(include="object"))

    st.subheader("Distribution of Selling Price")

    df["log_price"] = np.log(df["selling_price"])

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        fig1 = px.histogram(
            df, x="selling_price", nbins=50,
            title="Selling price",
            template="plotly_dark"
        )
        fig1.update_layout(height=350, title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        fig2 = px.histogram(
            df, x="log_price", nbins=50,
            title="Log Selling Price",
            template="plotly_dark"
        )
        fig2.update_layout(height=350, title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation heatmap of numeric features")

    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr().round(2)

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Correlation Matrix",
    )
    fig_corr.update_layout(title_x=0.5, height=700)

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Numeric features vs Selling Price")

    numeric_cols = [
        "max_power", "torque_nm", "max_torque_rpm",
        "age", "km_per_year", "engine per liter",
        "power_per_liter", "torque_per_liter"
    ]

    for col in numeric_cols:
        st.markdown(f"Feature: **{col}**")

        col_a, col_b, col_c = st.columns([1, 4, 1])
        with col_b:
            fig = px.scatter(
                df, x=col, y="selling_price",
                template="plotly_dark",
                opacity=0.6
            )
            fig.update_layout(
                height=330,
                title=f"{col} vs Selling Price",
                title_x=0.5
            )
            fig.update_traces(marker=dict(size=4))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
elif menu == "Prediction":
    st.header("Car Price Prediction")

    preprocessor = model.named_steps["preprocess"]
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]

    mode = st.radio("Choose option:", 
                    ["Load CSV", "Enter custom features"],
                    horizontal=True)

    if mode == "Load CSV":
        st.subheader("Upload your CSV file")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file with car features", 
            type=["csv"],
            help="File should contain all required features"
        )

        if uploaded_file:
            try:
                df_input = pd.read_csv(uploaded_file)
                
                st.success(f"File loaded successfully! Rows: {df_input.shape[0]}, Columns: {df_input.shape[1]}")
                
                st.subheader("Data preview")
                st.dataframe(df_input.head(), use_container_width=True)
                
                missing_cols = [c for c in feature_names if c not in df_input.columns]
                
                if missing_cols:
                    st.error(f"Missing required features: {', '.join(missing_cols)}")
                    st.info(f"Required features: {', '.join(feature_names)}")
                else:
                    with st.spinner("Making predictions..."):
                        df_processed = prepare_features(df_input, feature_names)
                        y_pred_log = model.predict(df_processed)
                        df_input["predicted_price"] = np.exp(y_pred_log)
                    
                    st.subheader("Prediction Results")
                    
                    results_df = df_input[["predicted_price"] + list(feature_names)]
                    st.dataframe(results_df, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Min predicted price", f"{df_input['predicted_price'].min():,.0f}")
                    with col2:
                        st.metric("Max predicted price", f"{df_input['predicted_price'].max():,.0f}")
                    with col3:
                        st.metric("Average predicted price", f"{df_input['predicted_price'].mean():,.0f}")
                    
                    csv_out = df_input.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download predictions as CSV",
                        data=csv_out,
                        file_name="car_price_predictions.csv",
                        mime="text/csv",
                        help="Click to download predictions"
                    )
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    else:  
        st.subheader("Enter car features manually")
        
        with st.form("manual_input_form"):
            st.write("Enter values for each feature:")
            
            user_values = {}
            
            cols_per_row = 3
            features_list = list(feature_names)
            
            for i, col in enumerate(features_list):
                if i % cols_per_row == 0:
                    cols = st.columns(cols_per_row)
                
                with cols[i % cols_per_row]:
                    if col in num_cols:
                        val = st.number_input(
                            f"{col}",
                            value=0.0,
                            key=f"input_{col}"
                        )
                        user_values[col] = val
                    else:
                        
                        val = st.text_input(
                            f"{col}",
                            value="",
                            key=f"input_{col}"
                        )
                        user_values[col] = val
            
            predict_button = st.form_submit_button("🚗 Predict Price")
        
        if predict_button:
            missing_fields = [col for col in feature_names if user_values[col] == "" and col not in num_cols]
            
            if missing_fields:
                st.error(f"Please fill in all fields. Missing: {', '.join(missing_fields)}")
            else:
                with st.spinner("Predicting..."):
                    df_user = pd.DataFrame([user_values])
                    df_processed = prepare_features(df_user, feature_names)
                    pred_log = model.predict(df_processed)[0]
                    pred = np.exp(pred_log)
                
                st.success("Prediction complete!")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.subheader("Predicted Car Price")
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                        <h1 style='color: #2e86c1;'>₹{pred:,.0f}</h1>
                        <p style='color: #5d6d7e;'></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View input features"):
                        st.write("Entered values:")
                        input_df = pd.DataFrame([user_values]).T
                        input_df.columns = ["Value"]
                        st.dataframe(input_df, use_container_width=True)

elif menu == "Weights":
    st.header("Model Weights")

    try:
        preprocess = model.named_steps["preprocess"]
        ridge = model.named_steps["model"]

        num_cols = list(preprocess.transformers_[0][2])
        cat_cols = list(preprocess.transformers_[1][2])

        df = pd.read_csv("df_train.csv")

        num_weights = ridge.coef_[:len(num_cols)]

        weights_df = pd.DataFrame({
            "Feature": num_cols,
            "Weight": num_weights
        })

        start = len(num_cols)
        cat_weight_rows = []
        for col in cat_cols:
            unique_vals = df[col].nunique()
            end = start + unique_vals - 1  # because drop='first'

            coef_slice = ridge.coef_[start:end + 1]
            grouped_weight = coef_slice.mean()

            cat_weight_rows.append({
                "Feature": col,
                "Weight": grouped_weight
            })

            start = end + 1

        cat_df = pd.DataFrame(cat_weight_rows)

        total_df = pd.concat([weights_df, cat_df], ignore_index=True)
        total_df["Abs_Weight"] = total_df["Weight"].abs()
        total_df = total_df.sort_values("Abs_Weight", ascending=False)

        st.subheader("Sorted Feature Importance")
        st.dataframe(total_df, use_container_width=True)

        st.subheader("Top 15 Most Important Features")
        fig1 = px.bar(
            total_df.head(15),
            x="Feature", y="Weight",
            color="Weight",
            color_continuous_scale="Tealrose",
            title="Top 15 Coefficients (Original Features)"
        )
        fig1.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig1, use_container_width=True)

    except Exception as e:
        st.error(f"Model weight visualization failed: {e}")