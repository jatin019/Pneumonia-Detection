import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def image_prediction(new_image_path):
    # Load and preprocess the image
    test_image = image.load_img(new_image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0
    
    # Load the model
    model_loaded = tf.keras.models.load_model("my_pneumonia_detection_model.h5")
    
    # Make prediction
    prediction = model_loaded.predict(test_image)
    
    # Determine prediction details
    class_names = ['Normal', 'Pneumonia']
    predicted_class = class_names[1 if prediction[0][0] > 0.5 else 0]
    confidence = float(prediction[0][0] * 100 if predicted_class == 'Pneumonia' else (1.0 - prediction[0][0]) * 100)
    
    return predicted_class, confidence

# Streamlit specific implementation
def main():
    st.title('Pneumonia Detection')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a chest X-ray image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded X-ray Image', width=300)
        
        # Save the uploaded file temporarily
        with open("temp_image.jpeg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Make prediction
        predicted_class, confidence = image_prediction("temp_image.jpeg")
        
        # Display prediction
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")
        
        # Color-coded result
        if predicted_class == 'Pneumonia':
            st.error(f"⚠️ Pneumonia Detected with {confidence:.2f}% confidence")
        else:
            st.success(f"✅ Normal Chest X-ray with {confidence:.2f}% confidence")

if __name__ == '__main__':
    main()