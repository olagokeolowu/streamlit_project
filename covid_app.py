import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# create title for app

st.title("Covid-19 Image Classification")
st.write("Upload a Covid Image, and we will predict which it is")

# create a file uploader
uploaded_file = st.file_uploader("Upload an image..", type = ["jpg", "jpeg", "png"])

# check if the file is uploaded
if uploaded_file is not None:
    # display the image
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded Image")
    st.write("")

    # preprocess the image
    img = np.array(image)
    img = tf.image.resize(img, (64, 64))
    # normalize the image
    img = img / 255.0
    img = np.expand_dims(img, axis = 0)

    # load the trained model
    model = load_model("C:/Users/HP/Desktop/Project_1/Covid/covid3.hdf5")

    # make predictions
    prediction = model.predict(img)

    # Define class labels
    labels = ['Covid', 'Viral Pneumonia', 'Normal']

    # # display the prediction to the screen
    # st.write(f" ## Predicted Image is: {label}")
    # for i, prob in enumerate(prediction[0]):
    #     st.write(f'{label[i]}: {prob * 100:.2f}%')

    # Get the predicted label
    predicted_label_index = np.argmax(prediction)
    predicted_label = labels[predicted_label_index]
    
    # Display the predicted label
    st.write(f"## Predicted Image is: {predicted_label}")

