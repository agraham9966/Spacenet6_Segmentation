import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import random
import warnings
import datetime 

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Cropping2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import Dropout, Lambda, Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, Add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

import tensorflow as tf
#https://stackoverflow.com/questions/58971998/in-tensorflow-2-0-how-to-feed-tfrecord-data-to-keras-model

def dice_coef(y_true, y_pred, smooth = 1.0):
    #quicker computation than metrics function 
    y_true_f = K.flatten(y_true)
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def metrics(y_true, y_pred, smooth = 1.0, th=0.5):
    y_true_f = K.flatten(y_true)
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred = tf.cast(y_pred >= th, tf.float32)
    y_pred_f = K.flatten(y_pred)
    
    tp = K.sum(y_true_f * y_pred_f) 

    diff = y_pred_f - y_true_f
    fp, fn = 0, 0
    for num in diff: 
        if num < 0: 
            fn += 1
        if num > 0: 
            fp += 1

    f1 = (2 * tp + 1) / (2 * tp + fn + fp + 1)
    
    return f1, tp, fp, fn

def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def dice_logloss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.15 + dice_coef_loss(y_true, y_pred) * 0.85

def unet1(input_shape = (320,320,8)):
    s = Input(input_shape)
    #s = Lambda(lambda x: x / 2048) (inputs)
    c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
    c5 = BatchNormalization()(c5)
    

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
    c6 = BatchNormalization()(c6)


    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    #crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(c9) #added cropping
    c9 = BatchNormalization()(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    # ###########

    model = Model(s, outputs)
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_coef])
    #model.compile(optimizer='adam', loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef])
    model.compile(optimizer='adam', loss=dice_logloss, metrics=[dice_coef, 'binary_crossentropy'])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_coef])
    model.summary()

    '''
    To load custom model when adding custom metrics such as dice_coef, use following: 
    model = load_model(saved_model_path.h5, custom_objects={'dice_coef':dice_coef})
    ''' 

    return model 

def deploy_unet(X_train_path, y_train_path, experiment_id ='A1P5', input_shape=(256,256,8)): 
    batch_size=3
    X_train_split, y_train_split, X_val_split, y_val_split = train_test_split(X_train_path, y_train_path, split=0.75)

    #initiate model 
    model = unet1(input_shape=input_shape)

    # #reduces learning rate if your metric stops improving... can be added to callbacks 
    # reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
    #                                    patience=10, 
    #                                    verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)


    earlystopper = EarlyStopping(patience=15, verbose=1)
    checkpointer = ModelCheckpoint('unet1-model-' + experiment_id + '-256x256x8in.h5', verbose=1, save_best_only=True)
    
    #fit model 

    history = model.fit(batch_gen(X_train_split, y_train_split, batch_size=batch_size),
                                    steps_per_epoch = int(float(len(X_train_split))/float(batch_size)),
                                    validation_data=batch_gen(X_val_split, y_val_split, batch_size=batch_size),
                                    validation_steps = int(float(len(X_val_split))/float(batch_size)),
                                    epochs = 50, 
                                    callbacks=[earlystopper,checkpointer])
    
    #plot_history(path, history)

    return 

def batch_gen(X_names, y_names, batch_size=8):

    rows = 256 
    cols = 256 
    X_channels = 8
    y_channels = 1

    while True: 

        c = list(zip(X_names, y_names))
        random.shuffle(c)
        _X_names, _y_names = zip(*c)

        X_ims = np.zeros((batch_size, rows, cols, X_channels))
        y_ims = np.zeros((batch_size, rows, cols, y_channels))

        ii=0

        for X_name, y_name in zip(_X_names,_y_names): 
            X_ims[ii] = np.load(X_name)[:rows,:cols,:X_channels]
            y_ims[ii] = np.load(y_name)[:rows,:cols,:y_channels]

            ii+=1
            
            if ii>= batch_size: 
                yield(X_ims, y_ims)
                #reset for next batch 
                X_ims = np.zeros((batch_size, rows, cols, X_channels))
                y_ims = np.zeros((batch_size, rows, cols, y_channels))
                ii=0

    return 

def train_test_split(X_train_path, y_train_path, split=0.9): 

    X_names = os.listdir(X_train_path)
    y_names = os.listdir(y_train_path)

    X_names = [os.path.join(X_train_path, i) for i in X_names]
    y_names = [os.path.join(y_train_path, i) for i in y_names]

    random.Random(4).shuffle(X_names)
    random.Random(4).shuffle(y_names)

    n_samples = len(X_names)

    X_train_split = X_names[0:int(split*n_samples)] 
    y_train_split = y_names[0:int(split*n_samples)]
    X_val_split = X_names[int(split*n_samples):] 
    y_val_split = y_names[int(split*n_samples):] 

    return X_train_split, y_train_split, X_val_split, y_val_split



if __name__ == '__main__': 

    deploy_unet(sys.argv[1], sys.argv[2])