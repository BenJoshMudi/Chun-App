import numpy as np
import pickle
import streamlit as st

# Load the saved model
model_path = "C:/Users/mudia/OneDrive/Desktop/Churn/model"
try:
    loaded_model = pickle.load(open(model_path, 'rb'))
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Create a function for prediction
def expresso_prediction(input_data):
    try:
        # Changing the input data to numpy array
        input_data_as_num = np.asarray(input_data, dtype=float).reshape(1, -1)
        
        # Check the shape of the input data
        if input_data_as_num.shape[1] != len(input_data):
            raise ValueError(f"Expected {len(input_data)} features, but got {input_data_as_num.shape[1]}")
        
        # Make prediction
        prediction = loaded_model.predict(input_data_as_num)
        return prediction[0]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Streamlit app
st.title("Customer Churn Prediction")

# User inputs
st.header("Enter customer details:")

# Creating a form for user input
with st.form(key='customer_form'):
    montant = st.number_input("Montant", min_value=0.0, value=0.0)
    frequence_rech = st.number_input("Frequence Rech", min_value=0.0, value=0.0)
    revenue = st.number_input("Revenue", min_value=0.0, value=0.0)
    arpu_segment = st.number_input("ARPU Segment", min_value=0.0, value=0.0)
    frequence = st.number_input("Frequence", min_value=0.0, value=0.0)
    data_volume = st.number_input("Data Volume", min_value=0.0, value=0.0)
    on_net = st.number_input("On Net", min_value=0.0, value=0.0)
    orange = st.number_input("Orange", min_value=0.0, value=0.0)
    tigo = st.number_input("Tigo", min_value=0.0, value=0.0)
    zone1 = st.number_input("Zone1", min_value=0.0, value=0.0)
    zone2 = st.number_input("Zone2", min_value=0.0, value=0.0)
    regularity = st.number_input("Regularity", min_value=0.0, value=0.0)
    freq_top_pack = st.number_input("Freq Top Pack", min_value=0.0, value=0.0)
    
    submit_button = st.form_submit_button(label='Predict')

# Process the form inputs
if submit_button:
    user_input = [
        montant, frequence_rech, revenue, arpu_segment, frequence, data_volume, 
        on_net, orange, tigo, zone1, zone2, regularity, freq_top_pack
    ]
    
    prediction = expresso_prediction(user_input)
    
    if prediction is not None:
        if prediction == 1:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')
    else:
        st.write('Error in making prediction. Please check the inputs and try again.')
