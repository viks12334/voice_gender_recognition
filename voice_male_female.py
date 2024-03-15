#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import librosa
import numpy as np
from google.colab import drive
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the expected input parameters for the model at the script level
num_mfcc = 13  # Adjust this based on your model's architecture
max_pad_length = 173  # Adjust this based on your model's architecture

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Dataset folder
dataset_folder = '/content/drive/My Drive/voice_data'

# Label mapping
label_mapping = {
    "male": 0,
    "female": 1
}

# Function to extract and pad MFCC features
def extract_mfcc(audio_path, num_mfcc=num_mfcc, max_pad_length=max_pad_length):
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc)

    # Pad or truncate the mfccs features to have the same length
    if mfccs.shape[1] > max_pad_length:
        mfccs = mfccs[:, :max_pad_length]
    elif mfccs.shape[1] < max_pad_length:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, max_pad_length - mfccs.shape[1])), mode='constant')

    return mfccs

# Labeling data with MFCC features
mfcc_data = []
for folder in os.listdir(dataset_folder):
    if os.path.isdir(os.path.join(dataset_folder, folder)):
        label = label_mapping.get(folder)
        if label is not None:
            for filename in os.listdir(os.path.join(dataset_folder, folder)):
                if filename.endswith(".wav"):
                    filepath = os.path.join(dataset_folder, folder, filename)
                    mfccs = extract_mfcc(filepath)
                    mfcc_data.append((mfccs, label))

# Pad all MFCCs to have the same length
for i, (mfcc, label) in enumerate(mfcc_data):
    if mfcc.shape[1] < max_pad_length:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant')
    elif mfcc.shape[1] > max_pad_length:
        mfcc = mfcc[:, :max_pad_length]
    mfcc_data[i] = (mfcc, label)

# Splitting data into train and test sets
X = np.array([mfcc[0] for mfcc, _ in mfcc_data])
y = np.array([label for _, label in mfcc_data])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
def build_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Reshape the data to match the input shape expected by the model
X_train = X_train[..., np.newaxis]  # Adding a channel dimension
X_test = X_test[..., np.newaxis]    # Adding a channel dimension

# Build and train the model
model = build_model(input_shape=X_train.shape[1:])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Save the model
model_path = '/content/drive/My Drive/voice__fm_model.h5'
model.save(model_path)
print(f"Model saved to {model_path}")

# Function to predict gender from audio, consistent with the training process
def predict_gender(audio_path, model):
    mfccs = extract_mfcc(audio_path)
    mfccs = mfccs[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    prediction = model.predict(mfccs)
    return "Male" if prediction > 0.5 else "Female"

