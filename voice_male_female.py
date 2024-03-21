#!/usr/bin/env python
# coding: utf-8

#このコードはGoogle Colaboratoryで作成されています。

import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, Flatten, GlobalMaxPooling2D, MaxPooling2D
from google.colab import drive
import string

# Mount Google Drive
drive.mount('/content/drive')

# 音声データの処理
def preprocess_audio(file_path, max_pad_len=937):
    audio, sample_rate = librosa.load(file_path, sr=48000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, hop_length=512)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width < 0:
        mfccs = mfccs[:, :max_pad_len]
    else:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

# 文字へのエンコーディング
characters = ' ' + string.ascii_lowercase
num_classes = len(characters)
char_to_index = dict((c, i) for i, c in enumerate(characters))

def one_hot_encode(character):
    vector = np.zeros((num_classes,))
    if character in char_to_index:
        vector[char_to_index[character]] = 1
    return vector

def encode_transcription(transcription, max_length=100):
    encoded = np.zeros((max_length, num_classes))
    for i, character in enumerate(transcription[:max_length]):
        encoded[i] = one_hot_encode(character)
    return encoded

# 音声データの読み込み
def load_data(data_directory='/content/drive/My Drive/voice_data', max_length=100):
    data = []
    labels = ['male', 'female']
    for label in labels:
        path = os.path.join(data_directory, label)
        for filename in os.listdir(path):
            if filename.endswith('.wav'):
                file_path = os.path.join(path, filename)

                # ファイルが見つからない場合はダミーの文字を割り当てる
                transcription = "dummy transcription"

                # 文字のエンコード
                encoded_transcription = encode_transcription(transcription, max_length=max_length)

                data.append((file_path, label, encoded_transcription))
    return data

# モデルの作成
def create_model(input_shape, output_dim):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(2, 2), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(2, 2), activation='relu')(x)  # Additional Conv2D layer
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = GlobalMaxPooling2D()(x)  # Replace Flatten with GlobalMaxPooling2D
    x = tf.expand_dims(x, 1)  # Add the time dimension
    x = LSTM(128)(x)
    stt_branch = Dense(output_dim, activation='softmax', name='stt_output')(x)
    gender_branch = Dense(1, activation='sigmoid', name='gender_output')(x)
    model = Model(inputs=input_layer, outputs=[stt_branch, gender_branch])
    return model

# データの読み込みと前処理
data = load_data()
x_train = np.array([preprocess_audio(file_path) for file_path, _, _ in data])
x_train = np.expand_dims(x_train, -1)  # Add channel dimension

# ラベルの準備
y_train_gender = np.array([1 if label == 'male' else 0 for _, label, _ in data])
y_train_stt = np.array([transcription for _, _, transcription in data])

# モデルの学習
max_transcription_length = 100  # Adjust based on your dataset
model = create_model(input_shape=(40, 937, 1), output_dim=max_transcription_length * num_classes)
model.compile(optimizer='adam',
              loss={'stt_output': 'categorical_crossentropy', 'gender_output': 'binary_crossentropy'},
              metrics=['accuracy'])
model.fit(x_train, {'stt_output': y_train_stt.reshape(len(y_train_stt), -1), 'gender_output': y_train_gender}, epochs=10, batch_size=32)

# モデルの保存
model.save('/content/drive/My Drive/model.h5')
