import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.layers.core import Dropout
from keras import regularizers
from keras.layers import BatchNormalization



class anime_model:
	
	def build(rows, columns, channels, labels, activation='relu', weightsPath=None):
	
		model = Sequential()
	
		inputShape = (rows, columns, channels)
		
		if K.image_data_format() == 'channels_first':
			inputShape = (channels, rows, columns)
			
		#First Layer
		model.add(Conv2D(32, 3, padding='same', input_shape=inputShape, 		    
				activation=activation, use_bias=True))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		model.add(Dropout(0.25))
		
		#Second Layer
		model.add(Conv2D(64, 3, padding='same', activation=activation))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		model.add(Dropout(0.25))
		
		#Third Layer
		model.add(Conv2D(128, 3, padding='same', activation=activation))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		model.add(Dropout(0.25))
		
		
		#Flatten and FC Layer
		model.add(Flatten())
		model.add(Dense(512, activation=activation))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		
		#Output Layer
		model.add(Dense(labels, activation='softmax'))
		
		if weightsPath is not None:
			model.load_weights(weightsPath)
		
		return model
	
		





