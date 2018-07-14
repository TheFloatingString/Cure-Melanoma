import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Model, load_model
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

model = load_model('D:/Hackerfest2018/model-hackathon2018-2.h5')

photo = imread('D:/Hackerfest2018/Presentation/Dermatofibroma.png')[:,:,:IMG_CHANNELS]
photo = resize(photo, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
photo = np.expand_dims(photo, axis=0)
print(photo.shape)

prediction = model.predict(photo)
print(prediction)