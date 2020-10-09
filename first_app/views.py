from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages


def index(request):
    return render(request, 'index.html')



import numpy as np
import cv2
import os
import pywt
import matplotlib
from matplotlib import pyplot as plt
import joblib
import pickle
import json

import base64
import cv2


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))




########################### This is the most important part to load Deep Learning Network #############################

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)



################################### Load the Model #############################################

modeldir = os.path.join(BASE_DIR, 'static/ml_files/tomatto_model_best_one.h5')


import tensorflow as tf
from tensorflow.keras.models import load_model
loaded_model = load_model(modeldir)


#################################### Load the CV2 ##################################################

import matplotlib.image as mpimg
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def img_to_np(DIR,flatten=True):
  #canny edge detection by resizing
  cv_img=mpimg.imread(DIR,0)
  cv_img=cv2.resize(cv_img,default_image_size)
  img = np.uint8(cv_img)
  #img = np.uint8((0.2126 * img[:,:,0]) + np.uint8(0.7152 * img[:,:,1]) + np.uint8(0.0722 * img[:,:,2]))
  #flatten it
  if(flatten):
    img=img.flatten()
  return img


labels=[
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___healthy",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_mosaic_virus",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

labelencoder = LabelBinarizer()
label=labelencoder.fit_transform([0,1,2,3,4,5,6,7,8,9])
label

default_image_size = tuple((128,128))




from django.core.files.storage import FileSystemStorage
def uploadPic(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)

        imgdir = os.path.join(BASE_DIR, 'media', name)

        # imagedir = os.path.join(BASE_DIR, 'static/images/cifar10_images/airplane.jpg')


        arr=img_to_np(imgdir,flatten=False)
        arr=arr.reshape(1,128,128,3)
        disease = labels[labelencoder.inverse_transform(loaded_model.predict(arr))[0]]
        disease



########### for uploaded image dir ######################

        x = imgdir
        image_path = []

        for char in x:
            if char != '\\':
                image_path.append(char)
            else:
                image_path.append('/')

        final_img_path = ''.join(image_path)
        print(final_img_path)



        return render(request, 'index.html', {'winner' : disease, 'final_img_path' : final_img_path})
