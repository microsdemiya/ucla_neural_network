
import streamlit as st
from modules.model import load_model
import numpy as np

st.title("🧠 UCLA Neural Network Classifier")

model = load_model("models/nn_model.pkl")

f1 = st.slider("Feature 1", 0.0, 10.0, 5.0)
f2 = st.slider("Feature 2", 0.0, 10.0, 5.0)

if st.button("Classify"):
    input_data = np.array([[f1, f2]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Class: {np.argmax(prediction)}")
