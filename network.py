from dataLoader import DataLoader
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from custom_KFold import MyKFold


def create_model(img_size,
                 h1=32, h2=64, h3=128, h4=0,
                 embed_size=128, d1=64, d2=32,
                 drop_out_rate=0.1,
                 use_batch_norm=True):
    """ This functions defines initializes the model.
    """
    center_coords = keras.layers.Input(
        (2,), dtype='float32', name='center_coords')
    img = keras.layers.Input((img_size, img_size), dtype='float32', name='img')
    img_init = keras.layers.Input(
        shape=(img_size, img_size), dtype='float32', name='img_init')
    x = keras.layers.Reshape((img_size, img_size, 1))(img)
    x_init = keras.layers.Reshape((img_size, img_size, 1))(img_init)
    batch_1 = keras.layers.BatchNormalization()
    CNN_1 = keras.layers.Conv2D(
        filters=h1, kernel_size=3, activation=tf.nn.relu)
    pool_1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    if drop_out_rate > 0:
        drop1 = keras.layers.Dropout(rate=drop_out_rate)
        x = drop1(x)
        x_init = drop1(x_init)
    if use_batch_norm:
        x = batch_1(x)
        x_init = batch_1(x_init)
    x = CNN_1(x)
    x_init = CNN_1(x_init)
    x = pool_1(x)
    x_init = pool_1(x_init)
    if not h2 == 0:
        if drop_out_rate > 0:
            drop2 = keras.layers.Dropout(rate=drop_out_rate)
            x = drop2(x)
            x_init = drop2(x_init)
        CNN_2 = keras.layers.Conv2D(
            filters=h2, kernel_size=3, activation=tf.nn.relu)
        pool_2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        batch_2 = keras.layers.BatchNormalization()
        if use_batch_norm:
            x = batch_2(x)
            x_init = batch_2(x_init)
        x = CNN_2(x)
        x = pool_2(x)
        x_init = CNN_2(x_init)
        x_init = pool_2(x_init)
    if not h3 == 0:
        if drop_out_rate > 0:
            drop3 = keras.layers.Dropout(rate=drop_out_rate)
            x = drop3(x)
            x_init = drop3(x_init)
        CNN_3 = keras.layers.Conv2D(
            filters=h3, kernel_size=3, activation=tf.nn.relu)
        pool_3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        batch_3 = keras.layers.BatchNormalization()
        if use_batch_norm:
            x = batch_3(x)
            x_init = batch_3(x_init)
        x = CNN_3(x)
        x = pool_3(x)
        x_init = CNN_3(x_init)
        x_init = pool_3(x_init)
    if not h4 == 0:
        CNN_4 = keras.layers.Conv2D(
            filters=h4, kernel_size=3, activation=tf.nn.relu)
        pool_4 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        if use_batch_norm:
            batch_4 = keras.layers.BatchNormalization()
            x = batch_4(x)
            x_init = batch_4(x_init)
        x = CNN_4(x)
        x = pool_4(x)
        x_init = CNN_4(x_init)
        x_init = pool_4(x_init)
    if drop_out_rate > 0:
        drop4 = keras.layers.Dropout(rate=drop_out_rate)
        x = drop4(x)
        x_init = drop4(x_init)
    if use_batch_norm:
        batch_5 = keras.layers.BatchNormalization()
        x = batch_5(x)
        x_init = batch_5(x_init)
    x = keras.layers.Flatten()(x)
    x_init = keras.layers.Flatten()(x_init)
    embedLayer = keras.layers.Dense(embed_size)
    x = embedLayer(x)
    x_init = embedLayer(x_init)
    concat_flat = keras.layers.Concatenate()([x, x_init])
    dense1 = keras.layers.Dense(d1, activation=tf.nn.elu)(concat_flat)
    dense2 = keras.layers.Dense(d2, activation=tf.nn.elu)(dense1)
    out_u = keras.layers.Dense(2)(dense2)
    out = keras.layers.Add()([center_coords, out_u])
    # Wrap everythin in Keras model
    model = keras.Model(inputs=[img, img_init, center_coords], outputs=out)
    # Compile the model
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    return model
