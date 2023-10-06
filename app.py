import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# create title for app
st.title('Covid Image Classification')
st.write('Upload your X-ray Image of the Lung')

# create file uploader
uploaded_file = st.file_uploader('Upload an Image', type = ['jpg', 'jpeg', 'png', 'webp'])

# check if the image was uploaded
if uploaded_file is not None:
    
    # display the image
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    st.image(image, caption = 'Uploaded Image')
    st.write('')

    # preprocess the image
    img = np.array(image) # convert image to a numpy array

    # resize the image
    img = tf.image.resize(img, (128, 128))

    # normalize the image
    img = img / 255.0
    img = np.expand_dims(img, axis = 0)
    # st.write(f'{img.shape}')

    # load the trained model
    model = load_model('/Users/nj/Desktop/MLOps/Covid_image_classification/covid19_model.h5')

    # make predictions
    prediction = model.predict(img)
    # st.write(prediction)

    # determine the class with the highest probability
    predicted_label_index = np.argmax(prediction)

    # map predicted class index to corresponding label
    labels = ['Covid', 'Normal', 'Viral Pneumonia']
    predicted_label = labels[predicted_label_index]

    # display the preicted image
    st.write(f'Predicted Image: {predicted_label}')

    