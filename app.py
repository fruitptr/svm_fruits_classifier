import pickle
import streamlit as st
import cv2
from sklearn.preprocessing import StandardScaler
import numpy as np

model = pickle.load(open('fruits_classification_model2.pkl', 'rb'))

def main():
    st.title("Fruits Classification using SVM - ML Project")

    uploaded_image = st.file_uploader(label="Choose a file", type=['png', 'jpg'])
    if uploaded_image is not None:
        scaler = StandardScaler()
        image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
        image_array = np.array(image)
        scaled_image = scaler.fit_transform(image_array.flatten())
        predicted_class = model.predict(scaled_image)
        st.success('This is a {}'.format(predicted_class))

if __name__ == '__main__':
    main()