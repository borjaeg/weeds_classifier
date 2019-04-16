#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import os
import cv2
import base64
import pandas as pd

from keras.applications import *
from keras.models import Model, Input
from keras.layers import Dense, Flatten
from keras.layers import GlobalAvgPool2D
from keras.layers import GlobalMaxPool2D
from keras.layers import ZeroPadding2D, LeakyReLU
from keras.layers import Dropout, concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation

from keras import backend as K


from keras.optimizers import Adam, SGD

from sklearn.externals import joblib
#import pymysql.cursors

import warnings
import logging
logging.basicConfig(level=logging.DEBUG, format = "%(name)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")

RANDOM_STATE = 2019
print("MALAKA")
print(os.listdir("."))


# In[17]:


def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


# In[18]:


def preprocess_data(photo, color_mode, transfer_mode):
    
    if transfer_mode in [0,  4]:
        #im = cv2.resize(im, (224, 224))
        photo = cv2.resize(photo, (96, 96))
    elif transfer_mode in [1, 2]:
        #im = cv2.resize(im, (299, 299))
        photo = cv2.resize(im, (99, 99))
    elif transfer_mode == 3:
        #im = cv2.resize(im, (299, 299))
        photo = cv2.resize(photo, (99, 99))
    elif transfer_mode in [5, 6, 7, 8, 9, 10]:
        #im = cv2.resize(im, (299, 299))
        photo = cv2.resize(photo, (51, 51))
    else:
        logging.error("Exception occurred", exc_info=True)
        raise ValueError("Unknown transfer_mode")

    if color_mode == 0:
        pass
    elif color_mode == 1:
        photo = segment_plant(photo)
    elif color_mode == 2:
        photo = sharpen_image(segment_plant(photo))
    else:
        logging.error("Exception occurred", exc_info=True)
        raise ValueError("Unknown color_mode parameter")

    return photo


# In[19]:


def get_att_layer(att_mode, inp):
    if att_mode == 0:
        return Flatten(name=ATTENTION_NAME)(inp)
    elif att_mode == 1:
        return GlobalAvgPool2D(name=ATTENTION_NAME)(inp)
    elif att_mode == 2:
        return GlobalMaxPool2D(name=ATTENTION_NAME)(inp)
    elif att_mode == 3:
        x_avg = GlobalAvgPool2D()(inp)
        x_max = GlobalMaxPool2D()(inp)
        return concatenate([x_avg, x_max], name=ATTENTION_NAME)
    else:
        logging.error("Exception occurred", exc_info=True)
        raise ValueError("Unknown att_mode")


# In[20]:


def dense_set(inp_layer, n, activation, drop_rate=0.):
    dp = Dropout(drop_rate)(inp_layer)
    dns = Dense(n)(dp)
    bn = BatchNormalization(axis=-1)(dns)
    act = Activation(activation=activation)(bn)
    return act

# Conv. layers set
def conv_layer(feature_batch, feature_map, kernel_size=(3, 3),strides=(1,1), zp_flag=False):
    if zp_flag:
        zp = ZeroPadding2D((1,1))(feature_batch)
    else:
        zp = feature_batch
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(zp)
    bn = BatchNormalization(axis=3)(conv)
    act = LeakyReLU(1/10)(bn)
    return act


# In[21]:


def get_own_feature_extractor(X, y, initial_filter_mode, kernel_size_mode):
    inp_img = Input(shape=(X.shape[1], X.shape[2], X.shape[3]))

    if initial_filter_mode == 0:
        initial_filter_size = 64
    elif initial_filter_mode == 1:
        initial_filter_size = 32
    elif initial_filter_mode == 2:
        initial_filter_size = 128
    else:
        logging.exception("Unknown Filter mode")
        raise ValueError("Unknown Filter mode")
        
    if kernel_size_mode == 0:
        kernel_size = (3,3)
    elif kernel_size_mode == 1:
        kernel_size = (5,5)
    else:
        logging.exception("Unknown KS mode")
        raise ValueError("Unknown KS mode")
    # 51
    conv1 = conv_layer(inp_img, initial_filter_size, kernel_size, zp_flag=False)
    conv2 = conv_layer(conv1, initial_filter_size, kernel_size, zp_flag=False)
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)
    # 23
    conv3 = conv_layer(mp1, initial_filter_size*2, kernel_size, zp_flag=False)
    conv4 = conv_layer(conv3, initial_filter_size*2, kernel_size, zp_flag=False)
    mp2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv4)
    # 9
    conv7 = conv_layer(mp2, initial_filter_size*4, kernel_size, zp_flag=False)
    conv8 = conv_layer(conv7, initial_filter_size*4, kernel_size, zp_flag=False)
    conv9 = conv_layer(conv8, initial_filter_size*4, kernel_size, zp_flag=False)
    mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv9)
    
    return Model(inputs=inp_img, outputs=mp3)


# In[22]:


def get_transfer_model(X, y, transfer_mode, att_mode, dense_unit, dense_dropout_rate, optimizer_mode):
    
    
    if transfer_mode == 0:
        feature_extractor = vgg19.VGG19(weights=WEIGHTS, include_top=False, 
                                        input_shape=(X.shape[1], X.shape[2], X.shape[3]))
        for layer in feature_extractor.layers:
            layer.trainable = False 

    elif transfer_mode == 1:
        feature_extractor = inception_v3.InceptionV3(weights=WEIGHTS, include_top=False,
                                                        input_shape=(X.shape[1], X.shape[2], X.shape[3]))
        for layer in feature_extractor.layers:
            layer.trainable = False
            
    elif transfer_mode == 2:
        feature_extractor = xception.Xception(weights=WEIGHTS, include_top=False,
                                                        input_shape=(X.shape[1], X.shape[2], X.shape[3]))
        for layer in feature_extractor.layers:
            layer.trainable = False
        
    elif transfer_mode == 3:
        feature_extractor = inception_resnet_v2.InceptionResNetV2(weights=WEIGHTS, include_top=False,
                                                                    input_shape=(X.shape[1], X.shape[2], X.shape[3]))
        for layer in feature_extractor.layers:
            layer.trainable = False
            
    elif transfer_mode == 4:
        feature_extractor = vgg19.VGG19(weights=WEIGHTS, include_top=False, 
                                        input_shape=(X.shape[1], X.shape[2], X.shape[3]))
        for layer in feature_extractor.layers[:FROZEN_LAYERS]:
            layer.trainable = False
            
    elif transfer_mode == 5:
        feature_extractor = get_own_feature_extractor(X, y, 0, 0)
    elif transfer_mode == 6:
        feature_extractor = get_own_feature_extractor(X, y, 0, 1)
    elif transfer_mode == 7:
        feature_extractor = get_own_feature_extractor(X, y, 1, 0)
    elif transfer_mode == 8:
        feature_extractor = get_own_feature_extractor(X, y, 1, 1)
    elif transfer_mode == 9:
        feature_extractor = get_own_feature_extractor(X, y, 2, 0)
    elif transfer_mode == 10:
        feature_extractor = get_own_feature_extractor(X, y, 2, 1)
    else:
        logging.error("Exception occurred", exc_info=True)
        raise ValueError("Unknown architecture")
            
    inp = feature_extractor.output
    x = get_att_layer(att_mode, inp)
    x = Dense(units=dense_unit, activation="tanh")(x)
    x = Dropout(rate=dense_dropout_rate)(x)
    out = Dense(units=len(y), activation="softmax")(x)

    model = Model(feature_extractor.input, out)
    
    # The first 50 epochs are used by Adam opt.
    # Then 30 epochs are used by SGD opt.
    
    if optimizer_mode == 0:
        mypotim = Adam(lr=2 * 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer_mode == 1:
        mypotim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)
    else:
        logging.error("Exception occurred", exc_info=True)
        raise ValueError("Unknown Optimizer") 
    
    
    model.compile(loss="categorical_crossentropy",
                  optimizer=mypotim,
                  metrics=["accuracy"])
    
    return model


# In[72]:


def load_architecture(photo, species, feature_extractor_name, transfer_mode, att_mode, 
                              dense_unit, dense_dropout_rate, image_generation, optimizer_mode):
    
    model = get_transfer_model(photo, species, transfer_mode, att_mode, dense_unit, 
                               dense_dropout_rate, optimizer_mode)
                                       
    model.load_weights(MODELS_PATH + feature_extractor_name)
    intermediate_layer_model = Model(inputs=model.input,
                                        outputs=model.get_layer(ATTENTION_NAME).output)
    
    return intermediate_layer_model


# In[82]:


def deep_extract_features(photo, feature_extractor):
    photo_features = feature_extractor.predict(np.expand_dims(photo, axis=0))

    return photo_features


# In[83]:

def get_best_model():
    METRIC_COLUMN = 14
    data = pd.read_csv(RESULTS_FILE, header=None)
    params = data.sort_values(by=[METRIC_COLUMN], ascending=False).head(1).values[0].tolist()
    name = ""
    for i, param in enumerate(params):
        if i not in [0,10,11,12,13,14]:
            name += str(param) + "_"
    
    return name[:-1]


def load_best_parameters(metric):

    # Connect to the database
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='conll',
                                 charset='utf8mb4')
    
    sql = """
        SELECT CONCAT_WS("_", color_mode, 
                    balance_mode, 
                    transfer_mode, 
                    att_mode, 
                    dense_unit, 
                    dense_dropout, 
                    optimizer_mode,
                    image_generation,
                    estimator) AS name
        FROM seed_classification_results
        ORDER BY {metric} DESC
        LIMIT 1
    """

    try:
    
        with connection.cursor() as cursor:
            cursor.execute(sql.replace("{metric}", metric))
            result = cursor.fetchone()
            print(result[0])
    finally:
        connection.close()


# In[ ]:

def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


#load_best_parameters("est_micro_f1")

ATTENTION_NAME = "attention"
WEIGHTS = "imagenet"
MODELS_PATH = "weeds_classifier/weeds_classifier/models/"
RESULTS_PATH = "weeds_classifier/weeds_classifier/results/"
RESULTS_FILE = RESULTS_PATH + "seeds_results_file_workshop.csv"

def analyse_photo(photo):
    K.clear_session()
    species = {0: 'Common Chickweed',
               1: 'Common wheat',
               2: 'Charlock',
               3: 'Loose Silky-bent',
               4: 'Maize',
               5: 'Sugar beet',
               6: 'Scentless Mayweed',
               7: 'Black-grass',
               8: 'Shepherds Purse',
               9: 'Cleavers',
               10: 'Fat Hen',
               11: 'Small-flowered Cranesbill'}
               
    
    feature_extractor_name = get_best_model()

    #feature_extractor_name = "0_0_9_3_256_0.15_0_1_SVM"
    logging.info("Loading... " + feature_extractor_name)
    arch_name = feature_extractor_name.split("_")
            
    color_mode = int(arch_name[0])
    balance_mode = int(arch_name[1])
    transfer_mode = int(arch_name[2])
    att_mode = int(arch_name[3])
    dense_unit = int(arch_name[4])
    dense_dropout_rate = float(arch_name[5])
    optimizer_mode = int(arch_name[7])
    image_generation = int(arch_name[6])
    estimator = arch_name[8]


    estimator_name = feature_extractor_name[:-3] + estimator + ".pkl"
    model = joblib.load(MODELS_PATH + estimator_name)
    
    photo = preprocess_data(photo, color_mode, transfer_mode)

    feature_extractor_name =  feature_extractor_name[:-4] + ".h5"

    deep_architecture = load_architecture(np.expand_dims(photo, axis=0), 
                                         species,
                                         feature_extractor_name, transfer_mode, att_mode, dense_unit, 
                                         dense_dropout_rate, image_generation, optimizer_mode)

    logging.info("Evaluating " + feature_extractor_name)

    features = deep_extract_features(photo, deep_architecture)
    
    prediction = model.predict(features)[0]
    logging.info(prediction)

    return species[prediction]
