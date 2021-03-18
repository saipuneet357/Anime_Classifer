import argparse
import cv2
from main import anime_model as nn
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.optimizers import SGD

def list_all_files(path):
	files = []
	
	for (r, d, f) in os.walk(path):
		
		for file in f:
			files.append(os.path.sep.join([r,file]))
	return files 


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

ap = argparse.ArgumentParser()

ap.add_argument('-d', '--dataset', required=True, help='path to dataset')
ap.add_argument('-w', '--weights', required=True, help='path to save weights')

args = vars(ap.parse_args())


filePaths = list_all_files(args['dataset'])


data = []
labels = []

#convert png to jpg later using imwrite

for filePath in filePaths:

	image = cv2.imread(filePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (100, 100))

	
	data.append(image)
	labels.append(filePath.split(os.path.sep)[-2])
	
	


trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

trainX = np.array(trainX, dtype='float32')/255.0
testX = np.array(testX, dtype='float32')/255.0

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)



if K.image_data_format() == 'channels_first':

	trainX = trainX.reshape((trainX.shape[0], 1, 100, 100))
	testX = testX.reshape((testX.shape[0], 1, 100, 100))
else:
	trainX = trainX.reshape((trainX.shape[0], 100, 100, 1))
	testX = testX.reshape((testX.shape[0], 100, 100, 1))
	
opt = SGD(lr=0.01)

model = nn.build(rows=100, columns=100, channels=1, labels=len(lb.classes_))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


print('[INFO] training..........')
model.fit(trainX, trainY, batch_size=32, epochs=20, verbose=1)

print('[INFO] evaluating...........')
(loss, accuracy) = model.evaluate(testX, testY, batch_size=128, verbose=1)
print('[INFO] accuracy: {}'.format(accuracy*100))

print('[INFO] dumping weights from model')
model.save_weights(args['weights'], overwrite=True)
#print(labels)



