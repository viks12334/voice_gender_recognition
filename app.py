#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import io

# Define the preprocess_audio function to handle the uploaded audio file
def preprocess_audio(uploaded_file, max_pad_len=937):
    # Convert the uploaded file to a BytesIO object
    file_bytes = io.BytesIO(uploaded_file.read())
    
    # Load the audio file with librosa
    audio, sample_rate = librosa.load(file_bytes, sr=48000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, hop_length=512)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width < 0:
        mfccs = mfccs[:, :max_pad_len]
    else:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

# Define a function to load the model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Streamlit web app
st.title("Audio Transcription and Gender Detection")

# Audio file uploader
audio_file = st.file_uploader("Upload audio", type=['wav'])

if audio_file is not None:
    # Process the audio file
    audio_data = preprocess_audio(audio_file)

    # Reshape the data to fit the model input
    audio_data = np.expand_dims(audio_data, axis=[0, -1])

    # Load the model
    model = load_model('./model.h5')

    # Make predictions
    stt_prediction, gender_prediction = model.predict(audio_data)

    # Decode the transcription and gender prediction
    # Note: You'll need to adapt this part based on how your model outputs the transcription and gender
    transcription = "Decoded transcription here"
    gender = "Male" if gender_prediction[0][0] > 0.5 else "Female"

    # Display the results
    st.write(f"Transcription: {transcription}")
    st.write(f"Gender: {gender}")
