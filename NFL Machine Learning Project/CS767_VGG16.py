
import numpy as np
import os
import time
from vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

#Import image data
PATH = os.getcwd()

#Locates the /data directory that contains the Pass/Run screenshots
#Pass and Run images are stored in their own respective sub-directory within /data
#sub-directory titles match the type of play
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

#Removes .DS_Store directory from the list of subdirectories
data_dir_list = data_dir_list[-2:]

img_data_list=[]

#Iterate over each subdirectory (pass or run) and perform pre-processing on each image
#Each image is then added to the img_data_list object
for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		if img =='.DS_Store':
			pass
		else:
			img_path = data_path + '/'+ dataset + '/'+ img
			img = image.load_img(img_path, target_size=(224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			#x = x/255 (can be used for normalization if necessary)
			print('Input image shape:', x.shape)
			img_data_list.append(x)


#Converts images to numpy arrays in the correct dimensions [batch, height, width, color channels]
img_data = np.array(img_data_list)
img_data=np.rollaxis(img_data,1,0)
img_data=img_data[0]

# Define the number of classes: 2 (Pass/Run)
num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

#Sets up labels for pass (0) and run images
#First 369 images are pass, last 366 are run (1)
labels[0:369]=0
labels[370:735]=1

names = ['Pass','Run']

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#Input layer for the model
image_input = Input(shape=(224, 224, 3))

#Imports the pre-trained VGG16 model with imagenet weights, top layers are included
model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()

#Connects the output of the pre-trained model to a new fully connected dense layer with 2 classes
last_layer = model.get_layer('fc2').output
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

#Prevents all layers except the new 2 node output layer from having weights altered during training
#This maintains the pre-trained imagenet weights of the VGG16 model. 
for layer in custom_vgg_model.layers[:-1]:
	layer.trainable = False

#Compiles the model with the Adam optimizer
custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Creates a checkpoint for the model weights that will only be updated whenever the model 
#experiences an improvement in validation accuracy
filepath='best.weights_Adamv3.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#Fits and evalutes the model with input batches of size 10
#Training occurs for 20 epochs, monitoring training loss, accuracy, validation loss and accuracy
t=time.time()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=10, callbacks=callbacks_list, epochs=20, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))



#Produces graphical visualizations of the model's performance
import matplotlib.pyplot as plt

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(20)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

#Charts the model's loss values over the 20 epochs
plt.show()

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#Charts the model's accuracy values over the 20 epochs.
plt.show()
