import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    trained_model = pickle.load(model_file)

st.title('Titanic Survival Prediction')

st.write("Note")
st.write("Please give input based on the changes we made in the data")

# Get numerical input features from the user
numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
input_features = []

for feature_name in numerical_features:
    value = st.number_input(f"Enter value for {feature_name}: ")
    input_features.append(value)

# Get categorical input features from the user
embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
sex_dict = {'male': 1, 'female': 0}

feature_texts = ['Sex', 'Embarked']

for feature_name in feature_texts:
    value = st.text_input(f"Enter value for {feature_name}: ")

    if feature_name == 'Sex':
        value = sex_dict.get(value)
        if value is not None:
            input_features.append(value)
        else:
            st.error(f"Invalid input for {feature_name}. Please enter 'male' or 'female'.")

    if feature_name == 'Embarked':
        value = embarked_dict.get(value)
        if value is not None:
            input_features.append(value)
        else:
            st.error(f"Invalid input for {feature_name}. Please enter 'C', 'Q', or 'S'.")

# Add a prediction button
survival_dict={1:"You will/have survived", 0:"You didn't survive"}
if st.button('Predict'):
    # Convert input features to a NumPy array
    input_features_array = np.array(input_features).reshape(1, -1)

    # Make predictions using the loaded model
    predicted_class = trained_model.predict(input_features_array)

    st.write(f"Predicted class: {survival_dict.get(predicted_class[0])}")
