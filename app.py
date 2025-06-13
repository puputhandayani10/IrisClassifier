import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.title("Klasifikasi Bunga Iris")

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Input fitur
sepal_length = st.slider('Sepal Length', float(X.iloc[:, 0].min()), float(X.iloc[:, 0].max()))
sepal_width = st.slider('Sepal Width', float(X.iloc[:, 1].min()), float(X.iloc[:, 1].max()))
petal_length = st.slider('Petal Length', float(X.iloc[:, 2].min()), float(X.iloc[:, 2].max()))
petal_width = st.slider('Petal Width', float(X.iloc[:, 3].min()), float(X.iloc[:, 3].max()))

# Klasifikasi
model = RandomForestClassifier()
model.fit(X, y)
pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

st.subheader("Hasil Prediksi")
st.write("Jenis bunga iris:", iris.target_names[pred[0]])
