import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤",
    layout="wide"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    
    .stApp {
        background-color: #f0f2f6;
    }

    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    
    .stButton > button {
        background-color: #007bff; /* Blue color */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        color: white;
    }

    
    .stAlert {
        border-radius: 8px;
    }
    
    
    [data-testid="stSuccess"] {
        background-color: #d4edda;
        color: #155724;
    }

    
    [data-testid="stError"] {
        background-color: #f790;
        color: #721c24;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a1a1a;
    }

    </style>
    """, unsafe_allow_html=True)

# --- Model Training ---
@st.cache_data  # Cache the data loading and model training
def load_data_and_train_model():
    try:
        df = pd.read_csv('heart.csv')
    except FileNotFoundError:
        st.error("Error: 'heart.csv' file not found.")
        st.info("Please make sure 'heart.csv' is in the same directory as this 'app.py' file.")
        return None, None
    
    # Define features (X) and target (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return df, model

df, model = load_data_and_train_model()

if df is None:
    st.stop()

# Calculate model accuracy on the full dataset
model_accuracy = accuracy_score(df.iloc[:, -1], model.predict(df.iloc[:, :-1])) * 100

# --- Sidebar for User Input ---
st.sidebar.header('Patient Input Features')
st.sidebar.markdown('Enter the patient data below to get a prediction.')

# Get feature names from the dataframe
feature_names = df.iloc[:, :-1].columns.tolist()

# Use sliders for continuous features and selectbox for categorical ones
with st.sidebar:
    age = st.slider('Age', 29, 77, 54)
    sex = st.selectbox('Sex', (0, 1), format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.selectbox('Chest Pain Type (cp)', (0, 1, 2, 3))
    trestbps = st.slider('Resting Blood Pressure (trestbps)', 94, 200, 131)
    chol = st.slider('Serum Cholesterol (chol)', 126, 564, 246)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1))
    restecg = st.selectbox('Resting ECG (restecg)', (0, 1, 2))
    thalach = st.slider('Max Heart Rate (thalach)', 71, 202, 149)
    exang = st.selectbox('Exercise Induced Angina (exang)', (0, 1))
    oldpeak = st.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0)
    slope = st.selectbox('Slope of Peak ST (slope)', (0, 1, 2))
    ca = st.selectbox('Num Major Vessels (ca)', (0, 1, 2, 3, 4))
    thal = st.selectbox('Thalassemia (thal)', (0, 1, 2, 3))

# --- Main Page for Output ---
st.title('❤ Heart Disease Prediction Model')
st.write('This app uses the Logistic Regression model from your notebook to predict heart disease.')

# Collect inputs into a dictionary
input_data = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

# Create a numpy array in the same order as model was trained
input_array = np.array([input_data[name] for name in feature_names]).reshape(1, -1)

# --- Prediction Logic and Display ---
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Your Input:")
    # Display the inputs as a dataframe
    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df, hide_index=True)

    if st.button('Predict', type="primary", use_container_width=True):
        # Make prediction
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)

        # Get the probability of the predicted class
        probability_of_prediction = prediction_proba[0][prediction[0]] * 100

        if prediction[0] == 1:
            st.error(f"*Prediction: Patient HAS Heart Disease*")
            st.write(f"Confidence: *{probability_of_prediction:.2f}%*")
        else:
            st.success(f"*Prediction: Patient DOES NOT Have Heart Disease*")
            st.write(f"Confidence: *{probability_of_prediction:.2f}%*")

with col2:
    st.subheader("Model Information")
    st.write(f"*Model Accuracy:* {model_accuracy:.2f}% (on training data)")
    st.write("This app uses a *Logistic Regression model* trained on the full heart.csv dataset.")
    st.write("---")
    st.subheader("Training Data Overview (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)