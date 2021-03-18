import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class anime_model:
	
	def build(rows, columns, channels, labels, activation='relu', weightsPath=None):
	
		model = Sequential()
	
		inputShape = (rows, columns, channels)
		
		if K.image_data_format() == 'channels_first':
			inputShape = (channels, rows, columns)
			
		#First Layer
		model.add(Conv2D(32, 5, padding='same', input_shape=inputShape, 		    
				activation=activation))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		
		#Second Layer
		model.add(Conv2D(64, 5, padding='same', input_shape=inputShape, 		    
				activation=activation))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))		
		
		#Flatten and FC Layer
		model.add(Flatten())
		model.add(Dense(1000, activation=activation))
		
		#Output Layer
		model.add(Dense(labels, activation='softmax'))
		
		if weightsPath is not None:
			model.load_weights(weightsPath)
		
		return model
	
		
	








