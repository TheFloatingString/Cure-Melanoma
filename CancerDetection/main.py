from flask import Flask, render_template, request
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES

import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Model, load_model, Sequential
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


# Configure Flask app
app = Flask(__name__)
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)


# Keras model loading
model = Sequential()
model = load_model('model.h5')

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Read pandas dataframe
df_treatment = pd.read_csv("ref_treatment.csv", delimiter='|')

# List of possible diseases
disease_dict = {0: "Melanoma",
                1: "Melanocytic Nevus",
                2: "Basal Cell Carcinoma",
                3: "Pigmented Bowen's",
                4: "Pigmented Benign Keratoses",
                5: "Dermatofibroma",
                6: "Vascular Lesion"}


@app.route('/home')
def student():
   return render_template('student.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])

        # Get latest file upload
        list_of_files = glob.glob('C:\\Users\\laure\\Desktop\\Learning\\FlaskTutorial\\static\\img\\*')      # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        last_file = latest_file.split('\\')
        last_file = last_file[-1]
        print(latest_file)

        # pathname of last file
        last_file = "static\\img\\"+last_file

        photo = imread(last_file)[:,:,:IMG_CHANNELS]
        photo = resize(photo, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
        photo = np.expand_dims(photo, axis=0)
        print(photo.shape)

        # prediction
        prediction = model.predict(photo)
        prediction = prediction.tolist()
        print(prediction)
        index = prediction[0].index(1.0)
        print(index)
        predicted_disease = disease_dict[index]

        # recommended treatment
        row = df_treatment.loc[df_treatment['Disease'] == predicted_disease]
        treatment_description = row.Treatment.values[0]
        treatment_source = row.Source.values[0]
        treatment_url = row.URL.values[0]

        return render_template('analyze.html',
                                name=last_file,
                                diagnostics=predicted_disease,
                                treatment_description=treatment_description,
                                treatment_source=treatment_source,
                                treatment_url=treatment_url)

    return render_template('upload.html')



if __name__ == '__main__':
    app.run()
