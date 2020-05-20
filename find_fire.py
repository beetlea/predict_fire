from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
number = 0

#dataval_X = np.zeros((448,224,224,3))
#dataval_Y = np.zeros((448,4))
def count_pic(folder):
    return glob.glob(folder +"/*")

count_train  = len(count_pic("data/train/fire"))
count_train = count_train + len(count_pic("data/train/no_fire"))
count_val = len(count_pic("data/val/fire"))
count_val = count_val + len(count_pic("data/val/no_fire"))
dataset_X = np.zeros((count_train,224,224,3))####5167####363
dataset_Y = np.zeros((count_train,1))
dataval_X = np.zeros((count_val,224,224,3))
dataval_Y = np.zeros((count_val,1))
print(dataset_Y[0])

def data_load_image(folder, y, z):
    imagePath = folder + '/'
    jpg1 = ".jpg"
    y = int(y)
    global number
    global dataset_X
    global dataset_Y
    global dataval_X
    global dataval_Y
    for file in glob.glob(folder +"/*"):
        identify = os.path.splitext(os.path.basename(file))[0]
        #print (identify)
        cv = imagePath + identify + jpg1
        #print (cv)
        img = (cv2.imread(cv))
        img = cv2.resize(img,(224,224))
        #img = np.reshape(img,(3,224,224))
        if z == 1:
            dataset_X[number, :, : , :] = img/255.0
            dataset_Y[number, 0] = y
        else:
            dataval_X[number, :, :, :] = img/255.0
            dataval_Y[number, 0] = y
        number = number + 1

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    z = np.arange(a.shape[0])
    np.random.shuffle(z)
    print(z.shape)
    return a[z,:], b[z,:]


img_width, img_height = 224, 224
train_data_dir = "data/train"
validation_data_dir = "data/val"
nb_train_samples = 10
nb_validation_samples = 448
batch_size = 2
epochs = 50
img = Input(shape = (224, 224, 3))

#model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
#model = applications.mobilenet_v2.MobileNetV2(input_shape = (img_width, img_height, 3), include_top=False, weights='imagenet')
model = applications.ResNet50(
    weights = 'imagenet',include_top = False,
    input_tensor = img, input_shape = None, pooling = 'avg')

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers:
    layer.trainable = False

#Adding custom Layers 
x = model.layers[-1].output
#x = Flatten()(x)
#x = Dense(128, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(64, activation="relu")(x)

predictions = Dense(1, activation="sigmoid")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)
data_load_image("data/train/fire", 1, 1)
data_load_image("data/train/no_fire", 0, 1)

number = 0
data_load_image("data/val/fire", 1, 0)
data_load_image("data/val/no_fire", 0, 0)


dataset_X, dataset_Y = unison_shuffled_copies(dataset_X,dataset_Y)
dataval_X, dataval_Y = unison_shuffled_copies(dataval_X,dataval_Y)
# compile the model 
model_final.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
for i in range(6):
	print('STEP ' + str(i))
	model_final.fit(dataset_X, dataset_Y, epochs=1, batch_size=9, validation_data=(dataval_X, dataval_Y))
	dataset_X, dataset_Y = unison_shuffled_copies(dataset_X,dataset_Y)
	#model_final.fit(dataval_X, dataval_Y, epochs=1, batch_size=15)
	#dataval_X, dataval_Y = unison_shuffled_copies(dataval_X,dataval_Y)
	
model_final.save('model_resnet50.h5')
'''''
number = 0
data_load_image("data/train/borsh3", 0, 0)
data_load_image("data/train/chai3", 1, 0)
data_load_image("data/train/kofe3", 2, 0)
data_load_image("data/train/pel3", 3, 0)
_, accuracy = model_final.evaluate(dataval_X, dataval_Y)
print('Accuracy: %.2f' % (accuracy*100))
for i in range(5):
	print('STEP ' + str(i))
	model_final.fit(dataval_X, dataval_Y, epochs=1, batch_size=5)
	dataval_X, dataval_Y = unison_shuffled_copies(dataval_X,dataval_Y)
#_, accuracy = model_final.evaluate(dataval_X, dataval_Y)
#print('Accuracy: %.2f' % (accuracy*100))
#model_final.fit(dataval_X, dataval_Y, epochs=50, batch_size=10)
model_final.save('model_100+50_epochs.h5')
'''