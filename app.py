#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import io
import string

# Character set used during training
characters = ' ' + string.ascii_lowercase
num_classes = len(characters)

# Decoding function
def decode_transcription(prediction):
    # Assuming the prediction is a 2D array (time steps, num_classes)
    transcription = ''
    for timestep in prediction:
        char_index = np.argmax(timestep)
        transcription += characters[char_index]
    return transcription.strip()

# Audio processing function
def preprocess_audio(uploaded_file, max_pad_len=937):
    file_bytes = io.BytesIO(uploaded_file.read())
    audio, sample_rate = librosa.load(file_bytes, sr=48000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, hop_length=512)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width < 0:
        mfccs = mfccs[:, :max_pad_len]
    else:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

# Model loading function
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Streamlit app
st.title("Audio Transcription and Gender Detection")

audio_file = st.file_uploader("Upload audio", type=['wav'])

if audio_file is not None:
    audio_data = preprocess_audio(audio_file)
    audio_data = np.expand_dims(audio_data, axis=[0, -1])
    
    model = load_model('/path/to/your/model.h5')
    
    stt_prediction, gender_prediction = model.predict(audio_data)

    transcription = decode_transcription(stt_prediction[0])
    gender = "Male" if gender_prediction[0][0] > 0.5 else "Female"

    st.write(f"Transcription: {transcription}")
    st.write(f"Gender: {gender}")
