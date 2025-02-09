import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

# Define Logistic Regression Model for MPG
class LogisticRegressionModelMPG(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModelMPG, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Define Logistic Regression Model for Car Evaluation
class LogisticRegressionModelCar(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModelCar, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Function to scale features for MPG
def scale_features(input_features, mean, scale):
    input_features = np.array(input_features)
    scaled_features = (input_features - mean) / scale
    return scaled_features.tolist()

# Load model and preprocessing for MPG
@st.cache_resource
def load_mpg_model():
    checkpoint = torch.load('mpg_model_with_scaler.pth', weights_only=False)
    mean = checkpoint['scaler_mean']
    scale = checkpoint['scaler_scale']
    input_dim = len(checkpoint['features'])
    model = LogisticRegressionModelMPG(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, mean, scale, checkpoint['features']

# Load model and preprocessing for Car Evaluation
@st.cache_resource
def load_car_model():
    preprocessing_objects = joblib.load('preprocessing_objects.joblib')
    input_dim = len(preprocessing_objects['label_encoders'])
    output_dim = len(preprocessing_objects['target_encoder'].classes_)
    model = LogisticRegressionModelCar(input_dim, output_dim)
    model.load_state_dict(torch.load('pytorch_car_model.pth'))
    model.eval()
    return model, preprocessing_objects

# Preprocess input for Car Evaluation
def preprocess_car_input(input_data, preprocessing_objects):
    label_encoders = preprocessing_objects['label_encoders']
    input_df = pd.DataFrame([input_data])
    for feature, encoder in label_encoders.items():
        input_df[feature] = encoder.transform(input_df[feature])
    return torch.tensor(input_df.values, dtype=torch.float32)

# MPG Prediction Function
def predict_mpg(input_features):
    model, mean, scale, feature_names = load_mpg_model()
    scaled_features = scale_features(input_features, mean, scale)
    input_tensor = torch.tensor([scaled_features], dtype=torch.float32)
    with torch.no_grad():
        predicted_probabilities = model(input_tensor).item()
        predicted_label = 1 if predicted_probabilities >= 0.5 else 0
    return predicted_label, predicted_probabilities

# Car Evaluation Prediction Function
def predict_car(input_data):
    model, preprocessing_objects = load_car_model()
    processed_input = preprocess_car_input(input_data, preprocessing_objects)
    with torch.no_grad():
        prediction = model(processed_input)
        predicted_class = torch.argmax(prediction, dim=1).item()
        class_label = preprocessing_objects['target_encoder'].inverse_transform([predicted_class])[0]
    return class_label

# Streamlit App
def main():
    st.title("Unified Prediction App")
    st.sidebar.title("Choose Prediction Task")
    task = st.sidebar.selectbox("Select Task", ["MPG Prediction", "Car Evaluation"])

    if task == "MPG Prediction":
        st.header("MPG Prediction")
        model, _, _, feature_names = load_mpg_model()

        input_features = []
        for feature in feature_names:
            value = st.number_input(f"Enter value for {feature}:", value=0.0)
            input_features.append(value)

        if st.button("Predict MPG"):
            predicted_label, predicted_probabilities = predict_mpg(input_features)
            if predicted_label == 0:
                st.success(f"Predicted Label: {predicted_label} this car has low MPG. So the car doesn't help you save fuel")
                st.info(f"Predicted Probability: {predicted_probabilities:.4f}")
            elif predicted_label == 1:
                st.success(f"Predicted Label: {predicted_label} this car has high MPG. So the car help you save fuel")
                st.info(f"Predicted Probability: {predicted_probabilities:.4f}")

    elif task == "Car Evaluation":
        st.header("Car Evaluation")
        buying = st.selectbox("Buying Price", ['low', 'med', 'high', 'vhigh'])
        maint = st.selectbox("Maintenance Cost", ['low', 'med', 'high', 'vhigh'])
        doors = st.selectbox("Number of Doors", ['2', '3', '4', '5more'])
        persons = st.selectbox("Number of Persons", ['2', '4', 'more'])
        lug_boot = st.selectbox("Luggage Boot Size", ['small', 'med', 'big'])
        safety = st.selectbox("Safety", ['low', 'med', 'high'])

        input_data = {
            'buying': buying,
            'maint': maint,
            'doors': doors,
            'persons': persons,
            'lug_boot': lug_boot,
            'safety': safety
        }

        if st.button("Predict Car Evaluation"):
            try:
                class_label = predict_car(input_data)
                if class_label == 'unacc':
                    meanings = 'unacceptable'
                elif class_label == 'vgood':
                    meanings = 'very good'
                elif class_label == 'acc':
                    meanings = 'acceptable'
                else:
                    meanings = ''
                
                st.success(f"The predicted evaluation class is: ({class_label}) {meanings}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
