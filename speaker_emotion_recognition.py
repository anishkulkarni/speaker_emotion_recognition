import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend
import matplotlib.pyplot as plt
import librosa
import wave
import librosa
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import rmsprop
from pydub import AudioSegment
import speakerDiarization as sd
import csv
import sys

def label(predictions):
	predictor = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'neutral', 5:'sadness', 6:'surprise',7:'anger', 8:'disgust', 9:'fear', 10:'happiness', 11:'neutral', 12:'sadness', 13:'surprise'}
	return predictor[predictions.tolist().index(np.max(predictions))]

def generate_features_LSTM(speaker_file):
	y, sr = librosa.load(speaker_file)
	mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
	mfccs = mfccs.reshape((1,40,1))
	return mfccs

def generate_features_conv(speaker_file):
	X, sample_rate = librosa.load(speaker_file, res_type='kaiser_fast',duration=2.5 ,sr=44100,offset=0.5)
	sample_rate = np.array(sample_rate)
	mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=13),axis=0)
	if(mfccs.shape[0]>216):
		mfccs=mfccs[:216]
	else:
		mfccs = np.pad(mfccs, (0, 216-mfccs.shape[0]), 'constant')
		mfccs = mfccs.reshape(1,216,1)
	return mfccs

def generate_features(speaker_file):
	mfccs_conv = generate_features_conv(speaker_file)
	mfccs_LSTM = generate_features_LSTM(speaker_file)
	prediction_conv = model_conv.predict(mfccs_conv)[0]
	prediction_LSTM = list(model_LSTM.predict(mfccs_LSTM)[0])
	predictor_transfer = {0:4, 1:4, 2:3, 3:5, 4:0, 5:2, 6:1, 7:6}
	prediction_LSTM_altered=[0,0,0,0,0,0,0]
	for i in range(8):
		prediction_LSTM_altered[predictor_transfer[i]] += prediction_LSTM[i]
	prediction_LSTM_altered = np.array(prediction_LSTM_altered)
	prediction_LSTM_altered = np.tile(prediction_LSTM_altered,2)
	prediction = (5*prediction_LSTM_altered + 3*prediction_conv)/8
	return prediction

def create_model_conv():
	model = Sequential()
	model.add(Conv1D(256, 8, padding='same',input_shape=(216,1)))
	model.add(Activation('relu'))
	model.add(Conv1D(256, 8, padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(MaxPooling1D(pool_size=(8)))
	model.add(Conv1D(128, 8, padding='same'))
	model.add(Activation('relu'))
	model.add(Conv1D(128, 8, padding='same'))
	model.add(Activation('relu'))
	model.add(Conv1D(128, 8, padding='same'))
	model.add(Activation('relu'))
	model.add(Conv1D(128, 8, padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(MaxPooling1D(pool_size=(8)))
	model.add(Conv1D(64, 8, padding='same'))
	model.add(Activation('relu'))
	model.add(Conv1D(64, 8, padding='same'))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(14))
	model.add(Activation('softmax'))
	opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
	model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
	return model

def create_model_LSTM():
	model = Sequential()
	model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
	model.add(Dense(64))
	model.add(Dropout(0.4))
	model.add(Activation('relu'))
	model.add(Dense(32))
	model.add(Dropout(0.4))
	model.add(Activation('relu'))
	model.add(Dense(8))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	return model

def predict_emotions(speaker_files):
	emotions = []
	for i in speaker_files:
		emotions.append(label(generate_features(i)))
	os.remove(speaker_files[0])
	os.remove(speaker_files[1])
	return emotions

def separate_speakers(audio_path,direc):
	speakerSlice = sd.main_sp(audio_path, embedding_per_second=1.2, overlap_rate=0.4)
	a,b=split(speakerSlice,audio_path)
	merge(a,b,direc)
	return ['speaker_1_'+direc,'speaker_2_'+direc]

def main():
	path='input'
	languages=['English']
	for language in languages:
		with open('output/'+language.upper()+'.csv','w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			writer.writerow(['Filename','Speaker','Emotion'])
			
		for direc in os.listdir(path+'/'+language):
			emotions = predict_emotions(separate_speakers(path+'/'+language+'/'+direc,direc))
			for i in range(len(emotions)):
				print('Speaker: '+str(i+1)+' ... Emotion: ',emotions[i])
			with open('output/'+language.upper()+'.csv','a') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')
				writer.writerow([direc,'1',emotions[0]])
				writer.writerow([direc,'2',emotions[1]])
				

def split(speakerSlice,audio_path):
	list=[]
	for i in speakerSlice.items():
		sum=0
		for j in i[1]:
			sum += (j['stop']-j['start'])
		list.append((i[0],sum))

	sortedlist = sorted(list,key=lambda x: x[1], reverse=True)
	longest = speakerSlice[sortedlist[0][0]]
	seclongest = speakerSlice[sortedlist[1][0]]
	x=0
	y=0
	for seg in longest:
		t1 = seg['start']
		t2 = seg['stop']
		newAudio = AudioSegment.from_wav(audio_path)
		newAudio = newAudio[t1:t2]
		newAudio.export('1.' + str(x) + '.wav', format="wav")
		x=x+1

	for seg2 in seclongest:
		t3 = seg2['start']
		t4 = seg2['stop']
		newAudio = AudioSegment.from_wav(audio_path)
		newAudio = newAudio[t3:t4]
		newAudio.export('2.' + str(y) + '.wav', format="wav")
		y=y+1
	x=x-1
	y=y-1
	return (x,y)

def merge(a,b,direc):
	combined = AudioSegment.empty()
	for i in range(a+1):
		audio=AudioSegment.from_wav('1.' + str(i) + '.wav')
		combined += audio
		os.remove('1.' + str(i) + '.wav')
	combined.export('speaker_1_'+direc,format='wav')
	combined = AudioSegment.empty()
	for i in range(b+1):
		audio=AudioSegment.from_wav('2.' + str(i) + '.wav')
		combined += audio
		os.remove('2.' + str(i) + '.wav')
	combined.export('speaker_2_'+direc,format='wav')

model_LSTM=create_model_LSTM()
model_LSTM.load_weights('LSTM_weights.h5')
model_conv=create_model_conv()
model_conv.load_weights('conv_weights.h5')

if __name__ == '__main__':
	main()