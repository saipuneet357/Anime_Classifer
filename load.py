import argparse
import cv2
import numpy as np
import pickle
from keras.models import load_model
from keras import backend as K
import os



def predict(**args):

	label_dict = {'anime-girl':'anime', 'cartoon-girl':'cartoon'}
	
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'	
	image = args['image']
	output = image.copy()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (120, 120))
	image = np.array(image, dtype='float32')/255	

	if K.image_data_format() == 'channels_first':
		
		image = image.reshape((1, 1, 120, 120))
		
	else:
		image = image.reshape((1, 120, 120, 1))
	
	
	print('[INFO] Loading Model and Label Binarizer...........')
	model = load_model(args['model'])

	with open(args['labels'], 'rb') as f:
		lb = pickle.loads(f.read())
		
	preds = model.predict(image)

	if preds[0][0] <0.5:
		i = 0 
	else:
		i = 1

	label = lb.classes_[i]
	label = label_dict[label]

	#text = "{}".format(label)

	#cv2.putText(output, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1
	#	, (0,0,255), 2)	

	return (output,label)
			


