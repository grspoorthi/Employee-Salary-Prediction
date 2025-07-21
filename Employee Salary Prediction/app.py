import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import base64
import mimetypes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# background image function
def set_background(image_file):
    full_path = os.path.abspath(image_file)
    if os.path.exists(image_file):
        mime_type, _ = mimetypes.guess_type(image_file)
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("data:{mime_type};base64,{encoded}");
                background-size:cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(f"Background image **not found** at: `{image_file}`")

# Set background image
bg_image_path = os.path.join(os.path.dirname(__file__), "bg.png")
set_background(bg_image_path)

# Load encoders and model
model = joblib.load(open("LinearModel.pkl", "rb"))
gender_encoder = joblib.load(open("gender_encoder.pkl", "rb"))
country_encoder = joblib.load(open("country_encoder.pkl", "rb"))
department_encoder = joblib.load(open("department_encoder.pkl", "rb"))

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Salary Prediction", "Model Prediction"])

# Define pages
def home():
    st.title("💼Employee Salary Prediction ")
    
    # Overview Section
    st.subheader("📌 Overview")
    st.markdown("""
    This application uses machine learning to predict employee salaries based on multiple factors like years of experience, job rating, location, and gender.
    It enables HR professionals and stakeholders to ensure fair, transparent, and data-driven compensation planning.
    """)

    # Key Features Section
    st.subheader("🔑 Key Features")
    st.markdown("""
    - 🎯 Predicts employee salary based on selected inputs
    - ⚙️ Powered by a pre-trained Linear Regression model
    - 🚀 Fast and interactive UI built with Streamlit
    - 🧩 Modular design for easy extension and analysis
    """)

    # Model Performance Section
    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
    ### Model: Linear Regression
    - **Training R² Score**: -0.0394
    - **Mean Absolute Error**: 8437.25
    - **Root Mean Squared Error**: 91247959.80
    """)
    
    with col2:
        st.markdown("""
    ### Model: Decision Tree
    - **Training R² Score**: -0.9121
    - **Mean Absolute Error**: 10793.85
    - **Root Mean Squared Error**: 167859614.77
    """)
        
def salary_prediction():

    st.title("📊Salary Prediction")

    age = st.slider("Enter the years at company", min_value=0, max_value=50, value=3)
    jobrate = st.number_input("Enter the job rate", min_value=0.0, step=0.1, value=4.0)
    country = st.selectbox("Select Country", country_encoder.classes_)
    gender = st.selectbox("Select Gender", gender_encoder.classes_)
    department = st.selectbox("Department", department_encoder.classes_)

    if st.button("Predict Salary"):
        data = np.array([
            age,
            gender_encoder.transform([gender])[0],
            jobrate,
            department_encoder.transform([department])[0],
            country_encoder.transform([country])[0]
        ]).reshape(1, -1)

        salary = model.predict(data)[0]
        st.success(f"💰 Predicted Salary: ₹{salary:,.2f}")

        # Download as text
        download_text = f"""Employee Salary Prediction Report

Years at Company: {age}
Job Rate: {jobrate}
Country: {country}
Gender: {gender}
Department: {department}
Predicted Salary: {salary}
"""
        st.download_button(
            label="📥 Download Salary Report",
            data=download_text,
            file_name="salary_prediction.txt",
            mime="text/plain"
        )

def model_prediction():
    st.title("🧠 Model Prediction")
    
    st.markdown("This section shows how well the model performs using test data.")

    # Load your test dataset
    data = pd.read_csv(r"C:\Users\91901\OneDrive\Desktop\Employee data.csv")

    # Encode columns using the same encoders used in training
    data["Gender_encoded"] = gender_encoder.transform(data["Gender"])
    data["Country_encoded"] = country_encoder.transform(data["Country"])
    data["Department_encoded"] = department_encoder.transform(data["Department"])

    X_test = data[["Years", "Job Rate", "Country_encoded", "Gender_encoded", "Department_encoded"]]
    y_test = data["Annual Salary"]
    
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Display metrics
    st.subheader("📈 Model Performance")
    st.write(f"**Mean Absolute Error (MAE):** ₹{mae:,.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** ₹{rmse:,.2f}")

    metrics = ['Mean Absolute Error', 'Root Mean Squared Error']
    linear_scores = [8437.25, 9515.24]
    tree_scores = [10793.85, 12987.54]

    df_plot = pd.DataFrame({
        'Evaluation Metric': metrics * 2,
        'Score': linear_scores + tree_scores,
        'Model': ['Linear Regression'] * 2 + ['Decision Tree'] * 2
    })

    # Bar plot for comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_plot, x='Evaluation Metric', y='Score', hue='Model', ax=ax)
    ax.set_title("Model Comparison (MAE & RMSE)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Evaluation Metric")
    ax.grid(True, linestyle='--', alpha=0.3)

    st.pyplot(fig)

# Page router
if page == "Home":
    home()
elif page == "Salary Prediction":
    salary_prediction()
elif page == "Model Prediction":
    model_prediction()
