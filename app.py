#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os
import tempfile

# Define the expected input parameters for the model
num_mfcc = 13  # Adjust this based on your model's architecture
max_pad_length = 173  # Adjust this based on your model's architecture

# Load and return the pre-trained model, and print its expected input shape
def load_pretrained_model():
    model_path = './voice__fm_model.h5'
    model = load_model(model_path)
    
    # Check if the model has layers and the first layer has an input shape attribute
    if model.layers and hasattr(model.layers[0], 'input_shape'):
        print(f"Model input shape: {model.layers[0].input_shape}")
    else:
        print("Unable to determine the input shape of the model's first layer.")
    
    return model

# Extract MFCC features from audio, ensuring they match the expected input shape
def extract_mfcc(audio_path, num_mfcc=num_mfcc, max_pad_length=max_pad_length):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc)
    if mfccs.shape[1] < max_pad_length:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, max_pad_length - mfccs.shape[1])), mode='constant')
    elif mfccs.shape[1] > max_pad_length:
        mfccs = mfccs[:, :max_pad_length]
    return mfccs

# Predict gender from the audio file, ensuring the MFCCs are correctly shaped
def predict_gender(audio_path, model):
    mfccs = extract_mfcc(audio_path)
    # Debugging: print the MFCC shape before and after reshaping
    print(f"MFCC shape before reshaping: {mfccs.shape}")
    mfccs = mfccs[np.newaxis, ..., np.newaxis]  # Reshape mfccs to 4D if necessary
    print(f"MFCC shape after reshaping: {mfccs.shape}")
    prediction = model.predict(mfccs)
    return "Male" if prediction > 0.5 else "Female"

# Load the pre-trained model
model = load_pretrained_model()

# Streamlit app title
st.title("Audio Gender Prediction")

# JavaScript function for audio recording in the browser
def audio_recorder():
    st.markdown("""
        <script>
        const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

        async function recordAudio() {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            let audioChunks = [];

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            const start = () => {
                audioChunks = [];
                mediaRecorder.start();
            };

            const stop = () =>
                new Promise(resolve => {
                    mediaRecorder.addEventListener("stop", () => {
                        const audioBlob = new Blob(audioChunks);
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audio = new Audio(audioUrl);
                        const play = () => audio.play();
                        resolve({ audioBlob, audioUrl, play });
                    });

                    mediaRecorder.stop();
                });

            return { start, stop };
        }

        (async () => {
            const recordButton = document.getElementById('record');
            const stopButton = document.getElementById('stop');
            const audioElement = document.getElementById('audio');
            const downloadLink = document.getElementById('download');
            let recorder = await recordAudio();

            recordButton.addEventListener("click", () => {
                recorder.start();
                recordButton.setAttribute("disabled", true);
                stopButton.removeAttribute("disabled");
            });

            stopButton.addEventListener("click", async () => {
                audio = await recorder.stop();
                audioElement.src = audio.audioUrl;
                downloadLink.href = audio.audioUrl;
                downloadLink.download = 'audio.webm';
                downloadLink.style.display = 'block';
                recordButton.removeAttribute("disabled");
                stopButton.setAttribute("disabled", true);
            });
        })();
        </script>
        <button id="record">Record</button>
        <button id="stop" disabled>Stop</button>
        <audio id="audio" controls></audio>
        <a id="download" style="display: none">Download</a>
        """, unsafe_allow_html=True)

audio_recorder()

# File uploader for the audio file and prediction display
uploaded_file = st.file_uploader("Or upload an audio file", type=["webm", "wav", "mp3", "ogg"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    prediction = predict_gender(tmp_file_path, model)
    st.write(f"Predicted Gender: {prediction}")

    os.remove(tmp_file_path)
