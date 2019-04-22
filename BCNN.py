#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import copy
from PIL import Image
import random
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
import numpy as np
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pandas as pd

from model_builder import buil_bcnn
#from lvcut_loader import DataLoader
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

os.environ['CUDA_VISIBLE_DEVICES']='0'


# In[22]:


data_dir = 'train.csv'
BACTH_SIZE = 16
EPOCH = 300
split_ratio = 45000

data_frame = pd.read_csv(data_dir)
print(data_frame.shape)
all_data = []
for i in range(data_frame.shape[0]):
    feature = data_frame.iloc[i, 1:-1]
    all_data.append([np.reshape(np.array(feature, dtype=np.float32), (32,32,3)), data_frame.iloc[i, -1]])


# In[23]:


random.shuffle(all_data)
random.shuffle(all_data)

train_data = all_data[:45000]
print(train_data[0][1])
validation_data = all_data[45000:]

encoder = LabelEncoder()


# In[24]:


#用于生成批次数据的fit_generator
def generate_batch_data_random(data, batch_size, encoder):
    data_num = len(data)
    loopcount = data_num // batch_size
    data_x = [data[i][0] for i in range(data_num)]
    data_y = [data[i][1] for i in range(data_num)]
    data_y = encoder.fit_transform(data_y)
    data_y = np_utils.to_categorical(data_y)
    idx = 0
    while True:
        if idx < loopcount:
            batch_x = data_x[idx*batch_size : (idx+1)*batch_size]
            batch_x = np.array(batch_x, dtype=np.float32)
            batch_x = (batch_x-127.5)/127.5
            batch_y = data_y[idx*batch_size : (idx+1)*batch_size]
            batch_y = np.array(batch_y, dtype=np.float32)
            idx += 1
            yield (batch_x, batch_y)
        else:
            idx = 0
            
def preprocess_input(x):
    m = [cv2.getRotationMatrix2D(center=(x.shape[1] / 2, x.shape[0] / 2), angle=0, scale=1.0),
         cv2.getRotationMatrix2D(center=(x.shape[1] / 2, x.shape[0] / 2), angle=90, scale=1.0),
         cv2.getRotationMatrix2D(center=(x.shape[1] / 2, x.shape[0] / 2), angle=180, scale=1.0),
         cv2.getRotationMatrix2D(center=(x.shape[1] / 2, x.shape[0] / 2), angle=270, scale=1.0)]
    angle = np.random.randint(4)
    x = cv2.warpAffine(x, m[angle], (x.shape[1], x.shape[0]))
    return (x-127.5)/127.5


# In[20]:


def train_model(
        name_optimizer='sgd',
        learning_rate=0.05,
        decay_learning_rate=1e-9,
        all_trainable=True,
        model_weights_path=None,
        no_class=10,
        batch_size=BACTH_SIZE,
        epoch=300,

        tensorboard_dir=None,
        checkpoint_dir=None
    ):
    '''Train or retrain model.

    Args:
        train_dir: train dataset directory.
        valid_dir: validation dataset directory.
        name_optimizer: optimizer method.
        learning_rate: learning rate.
        decay_learning_rate: learning rate decay.
        model_weights_path: path of keras model weights.
        no_class: number of prediction classes.
        batch_size: batch size.
        epoch: training epoch.

        tensorboard_dir: tensorboard logs directory.
            If None, dismiss it.
        checkpoint_dir: checkpoints directory.
            If None, dismiss it.

    Returns:
        Training history.
    '''

    model = buil_bcnn(
        all_trainable=all_trainable,
        no_class=no_class,
        name_optimizer=name_optimizer,
        learning_rate=learning_rate,
        decay_learning_rate=decay_learning_rate,
        name_activation='softmax',
        name_loss='categorical_crossentropy')

    if model_weights_path:
        model.load_weights(model_weights_path)

    # Callbacks
    callbacks = []
    if tensorboard_dir:
        cb_tersoboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=False)
        callbacks.append(cb_tersoboard)

    #if checkpoint_dir:
        #cb_checkpoint = ModelCheckpoint(
            #os.path.join(checkpoint_dir, 'model_{epoch:02d}-{val_acc:.3f}.h5'),
            #save_weights_only=True,
            #monitor='val_loss',
            #verbose=True)
        #callbacks.append(cb_checkpoint)

    cb_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        min_delta=1e-3)
    cb_stopper = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=10,
        verbose=0,
        mode='auto')
    callbacks += [cb_reducer, cb_stopper]

    # Train
    # save best model
    filepath = "./checkpoint/sgd-weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath=filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only='True',
        mode='max',
        period=1)
    callback_list = [cb_reducer, checkpoint]


    train_generator = generate_batch_data_random(train_data, BACTH_SIZE, encoder)
    validation_generator = generate_batch_data_random(validation_data, BACTH_SIZE, encoder) 

    #loader = DataLoader(datapath=data_dir)

    # use generator
    #datagen = loader.generate(batch_size)
    #iterations = loader.train_size // batch_size
    history = model.fit_generator(
        train_generator, validation_data=validation_generator, epochs=epoch,
        steps_per_epoch=split_ratio/ batch_size,
        validation_steps=(50000 - split_ratio) / batch_size,
        callbacks=callback_list
        )

    return history


# In[21]:


train_model()


# In[ ]:





# In[ ]:




