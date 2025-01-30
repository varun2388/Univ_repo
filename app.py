import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Build a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit app
st.title("Iris Flower Prediction App")

st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    sepal_width = st.sidebar.slider('Sepal width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    petal_length = st.sidebar.slider('Petal length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
    petal_width = st.sidebar.slider('Petal width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
