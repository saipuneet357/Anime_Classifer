import argparse
import cv2
from model import anime_model as nn
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.optimizers import Adam,SGD
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import pickle

def list_all_files(path):
	files = []
	
	for (r, d, f) in os.walk(path):
		
		for file in f:
			files.append(os.path.sep.join([r,file]))
	return files 

def print_summary(data):
	print(data)
	f.write(data+"\n")
	


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

ap = argparse.ArgumentParser()

ap.add_argument('-d', '--dataset', required=True, help='path to dataset')
ap.add_argument('-w', '--weights', default=None, help='path to save weights')
ap.add_argument('-l', '--load', default=None, help='path to load weights')
ap.add_argument('-b', '--bin', default=None, help='path to save labels')

args = vars(ap.parse_args())

print(args)
filePaths = list_all_files(args['dataset'])


data = []
labels = []

#convert png to jpg later using imwrite
print('Loading Data......')
for filePath in filePaths:

	image = cv2.imread(filePath)
	if image is None:
		print('{} not found'.format(filePath))
		continue
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (120, 120))
	
	
	data.append(image)
	labels.append(filePath.split(os.path.sep)[-2])


print('Loading Done.........')
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

trainX = np.array(trainX, dtype='float32')/255.0
testX = np.array(testX, dtype='float32')/255.0

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)



if K.image_data_format() == 'channels_first':

	trainX = trainX.reshape((trainX.shape[0], 1, 120, 120))
	testX = testX.reshape((testX.shape[0], 1, 120, 120))
else:
	trainX = trainX.reshape((trainX.shape[0], 120, 120, 1))
	testX = testX.reshape((testX.shape[0], 120, 120, 1))

#Data Augmentation
#aug = ImageDataGenerator()
print('[INFO] Performing on the fly augmentation.......')
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	fill_mode='nearest'
)

INIT_LR = 0.001
EPOCHS = 60
	
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)
#opt = Adam()


model = nn.build(rows=120, columns=120, channels=1, labels=len(lb.classes_), weightsPath=args['load'])

#Summary of model
f = open('params.txt','w+')
model.summary(print_fn=print_summary)
f.close()


model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

if args['load'] == None:
	print('[INFO] training..........')
	history = model.fit(x=aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=EPOCHS, verbose=2)
	

if args['weights'] != None:
	print('[INFO] dumping weights from model........')
	model.save_weights(args['weights'], overwrite=True)
	model.save('model.h5')
	f = open(args['bin'], 'wb')
	f.write(pickle.dumps(lb))
	f.close()
else:
	print('[INFO] weights will not be saved as path not provided........')



print('[INFO] evaluating...........')
(loss, accuracy) = model.evaluate(testX, testY, batch_size=128, verbose=2)
print('[INFO] accuracy: {}.........'.format(accuracy*100))

#plot accuracy results
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('model accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()

#plot loss results
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('model loss')
plt.legend()
plt.savefig('loss.png')
plt.show()






