import os
import librosa
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from python_speech_features import mfcc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Activation, Reshape, Permute
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM
from keras import optimizers
import keras.callbacks
from keras.callbacks import EarlyStopping
from keras.models import load_model
from frfft import *

#### Step 1: Data collection
df = pd.read_csv('noisy_ICA_label.csv')

# Loading data
path = 'ICA_dataset_noisy/'
audio_data = list()
for i in tqdm(range(df.shape[0])):
    audio_data.append(librosa.load(path+df['fname'].iloc[i]))
audio_data = np.array(audio_data)

# Put the loaded data into data frame
df['audio_waves'] = audio_data[:,0]
df['samplerate'] = audio_data[:,1]
df.head()

# Calculate the length of each audio file
bit_lengths = list()
for i in range(df.shape[0]):
    bit_lengths.append(len(df['audio_waves'].iloc[i]))
bit_lengths = np.array(bit_lengths)
df['bit_lengths'] = bit_lengths
df['second_lengths'] = df['bit_lengths']/df['samplerate']
df.head()


# Take only the audio with >= 2 seconds audio length
df = df[df['second_lengths'] >= 2.0]

# create a checkpoint
with open('audio_df.pickle', 'wb') as f:
    pickle.dump(df, f)

# load a checkpoint
with open('audio_df.pickle', 'rb') as f:
    df = pickle.load(f)

num_samples = len(df['fname'])

audio_waves=list();
audio_waves_comlumn = df['audio_waves'].to_numpy()
for i in range(num_samples):
    audio_waves.append(audio_waves_comlumn[i])

labels=list();
labels_comlumn = df['label'].to_numpy()
for i in range(num_samples):
    labels.append(labels_comlumn[i])


generated_audio_waves = np.array(audio_waves)
generated_audio_labels = np.array(labels)
generated_audio_labels=generated_audio_labels[:,np.newaxis]
print(generated_audio_waves.shape)
print(generated_audio_labels.shape)

#### Step 2: Features preprocessing
mfcc_features = list()
for i in tqdm(range(len(generated_audio_waves))):
    mfcc_features.append(mfcc_fft(generated_audio_waves[i]))
mfcc_features = np.array(mfcc_features)

# create a checkpoint
with open('/content/mfcc_fft_fetures.pickle', 'wb') as f:
  pickle.dump(mfcc_features, f)

# load a checkpoint
with open('/content/mfcc_fft_fetures.pickle', 'rb') as f:
    mfcc_features = pickle.load(f)
    
# Check the shape of raw wave and MFCC features
print(generated_audio_waves.shape)
print(mfcc_features.shape)

# Print a raw audio wave
plt.figure(figsize=(12,2))
plt.plot(generated_audio_waves[30])
plt.title(generated_audio_labels[30])
plt.show()

# Print the MFCC features of the audio wave
plt.figure(figsize=(12, 2))
plt.imshow(mfcc_features[30].T, cmap='hot')
plt.title(generated_audio_labels[30])
plt.show()

#### Step 3: Label preprocessing
# Label encoding
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(generated_audio_labels)

label_encoded = label_encoded[:, np.newaxis]
print(label_encoded)

# One hot encoding
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(label_encoded)
print(one_hot_encoded)

#### Step 4: Model training
X = mfcc_features
y = one_hot_encoded
X = (X-X.min())/(X.max()-X.min())

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.33)
# Defining input shape for the neural network
input_shape = (X_train.shape[1], X_train.shape[2], 1)

# Reshape X_train and X_test such that they are having the same shape as the input shape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
print(X_train.shape)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
print(X_val.shape)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print(X_test.shape)

# Constructing the neural network architecture
model = Sequential()

model.add(BatchNormalization(axis=1, input_shape=input_shape))
model.add(Conv2D(64, (3, 3), strides=(1, 1), 
    padding='same', input_shape=input_shape))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(4, 2),strides=(4,2),padding='same'))
model.add(Dropout(0.1))


model.add(Permute((2, 1, 3)))
resize_shape = model.output_shape[2] * model.output_shape[3]
model.add(Reshape((model.output_shape[1], resize_shape)))
model.add(GRU(32, return_sequences=True))
model.add(GRU(32, return_sequences=False))
model.add(Dropout(0.3))


model.add(Dense(5, activation='softmax'))
model.summary()
opt = optimizers.Adam(learning_rate=0.0001)
callback = EarlyStopping(monitor="val_acc",mode='max',patience=10,min_delta=0)
model.compile(loss='categorical_crossentropy', 
     optimizer=opt,
     metrics=['acc'])

# Training the model
history = model.fit(X_train, y_train, epochs=30, batch_size = 32,
                    validation_data=(X_val, y_val), callbacks=[callback])
model.save('CRNN.h5')

# Displaying loss values
plt.figure(figsize=(8,8))
plt.title('Loss Value')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
print('loss:', history.history['loss'][-1])
print('val_loss:', history.history['val_loss'][-1])
plt.show()

# Displaying accuracy scores
plt.figure(figsize=(8,8))
plt.title('Accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
print('acc:', history.history['acc'][-1])
print('val_acc:', history.history['val_acc'][-1])
plt.show()

#### Step 6: Model evaluation
trained_model = load_model('CRNN.h5')
predictions = trained_model.predict(X_test)

predictions = np.argmax(predictions, axis=1)
y_test_inv = one_hot_encoder.inverse_transform(y_test)

# Creating confusion matrix
cm = confusion_matrix(y_test_inv, predictions)
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

