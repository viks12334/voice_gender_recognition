#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

def preprocess_audio(file_path, max_pad_len=937):
    audio, sample_rate = librosa.load(file_path, sr=48000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, hop_length=512)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width < 0:
        mfccs = mfccs[:, :max_pad_len]
    else:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

st.title("Voice Transcription and Gender Detection")

# Audio file uploader
audio_file = st.file_uploader("Upload audio", type=['wav'])

if audio_file is not None:
    # Display a warning if the audio is longer than 10 seconds
    audio_length = librosa.get_duration(filename=audio_file)
    if audio_length > 10:
        st.warning("Please upload an audio file of 10 seconds or less.")
    else:
        # Process the audio file if it's the correct length
        audio_data = preprocess_audio(audio_file)

        # Reshape the data to fit the model input
        audio_data = np.expand_dims(audio_data, axis=[0, -1])

        # Load the model (make sure to replace 'model_path' with the actual path to your model)
        model = load_model('./model.h5')

        # Make predictions
        stt_prediction, gender_prediction = model.predict(audio_data)

        # Decode the transcription and gender prediction
        transcription = "Decoded transcription here"
        gender = "Male" if gender_prediction[0][0] > 0.5 else "Female"

        # Display the results
        st.write(f"Transcription: {transcription}")
        st.write(f"Gender: {gender}")
