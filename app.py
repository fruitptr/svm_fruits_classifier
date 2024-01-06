import pickle
import streamlit as st
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
from rembg import remove
from io import BytesIO

model = pickle.load(open('fruits_classification_model2.pkl', 'rb'))
scaler = pickle.load(open('scalerinstance.pkl', 'rb'))

# Dictionary to map indices to fruit names
fruit_mapping = {
    0: 'Banana',
    1: 'Coconut',
    2: 'Peach',
    3: 'Pineapple'
}

def remove_transparency(source, background_color):
    source_img = cv2.cvtColor(source[:,:,:3], cv2.COLOR_BGR2GRAY)
    source_mask = source[:,:,3]  * (1 / 255.0)

    background_mask = 1.0 - source_mask

    bg_part = (background_color * (1 / 255.0)) * (background_mask)
    source_part = (source_img * (1 / 255.0)) * (source_mask)

    return np.uint8(cv2.addWeighted(bg_part, 255.0, source_part, 255.0, 0.0))

def perform_inference(uploaded_image, remove_background):
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

    if remove_background:
        image = remove(image)
        trans_mask = image[:,:,3] == 0
        image = image.copy()
        image[trans_mask] = [255, 255, 255, 255]
        temp_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        new_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    else:
        new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(new_img, (100, 100))
    flattened_image = resized_image.flatten()
    scaled_image = scaler.transform([flattened_image])
    predicted_index = model.predict(scaled_image)[0]
    predicted_fruit = fruit_mapping.get(predicted_index, 'Unknown')

    return predicted_fruit, new_img

def main():
    st.title("Fruits Classification using SVM - ML Project")

    uploaded_image = st.file_uploader(label="Choose a file", type=['png', 'jpg'])
    remove_background = st.checkbox("Remove Background", value=False)

    if uploaded_image is not None:
        if st.button('Predict'):
            predicted_fruit, image = perform_inference(uploaded_image, remove_background)
            st.success(f'This is a {predicted_fruit}')
            st.image(image, caption='Uploaded Image')

if __name__ == '__main__':
    main()
