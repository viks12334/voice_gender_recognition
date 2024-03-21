#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import io
import string

# 学習で使用する文字セット
characters = ' ' + string.ascii_lowercase
num_classes = len(characters)

# デコード関数
def decode_transcription(prediction):
    transcription = ''
    for timestep in prediction:
        char_index = np.argmax(timestep)
        transcription += characters[char_index]
    return transcription.strip()

# 音を処理する関数
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

# モデルの読み込み関数
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# 音声認識でJuliusを使用する関数
def get_transcription_with_julius(audio_path):
    julius_command = f"julius -C your_julius_config_file.jconf -input rawfile -file {audio_path}"
    result = subprocess.run(julius_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    transcription_lines = [line for line in result.stdout.split('\n') if 'sentence1:' in line]
    transcription = ' '.join(line.split(' ')[2:] for line in transcription_lines).replace('▁', ' ').strip()

    return transcription

# Streamlitアプリ
st.title("Audio transcription and gender identification from voice")
st.write("Upload an audio file for transcription and gender identification.")

audio_file = st.file_uploader("Upload audio", type=['wav'])

if audio_file is not None:
    temp_audio_path = 'temp_audio.wav'
    with open(temp_audio_path, 'wb') as f:
        f.write(audio_file.read())

    # Juliusを使用した文字起こし
    transcription = get_transcription_with_julius(temp_audio_path)

    # 性別の判定
    audio_data = preprocess_audio(audio_file)
    audio_data = np.expand_dims(audio_data, axis=[0, -1])

    model = load_model('./model.h5')

    _, gender_prediction = model.predict(audio_data)
    gender = "Male" if gender_prediction[0][0] > 0.5 else "Female"

    st.write(f"Transcription: {transcription}")
    st.write(f"Gender: {gender}")

