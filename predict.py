import keras
from keras.models import load_model
import cv2
import numpy as np
model = load_model('model_resnet50.h5')
img = cv2.imread('pic.jpg')
img = cv2.resize(img,(224,224))
dataval = np.zeros((1,224,224,3))

dataval [0,:,:,:] = img
print(model.predict(dataval))
