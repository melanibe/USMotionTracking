import tensorflow as tf
from tensorflow import keras
from dataLoader import metrics_distance

def create_model(img_size,
                 h1=32, h2=64, h3=128,
                 embed_size=128, d1=256,
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
    print(x.shape)
    batch_1_1 = keras.layers.BatchNormalization()
    batch_1_2 = keras.layers.BatchNormalization()
    batch_1_3 = keras.layers.BatchNormalization()
    CNN_1_1 = keras.layers.Conv2D(
        filters=h1, kernel_size=3, activation=tf.nn.relu)
    CNN_1_2 = keras.layers.Conv2D(
        filters=h1, kernel_size=3, activation=tf.nn.relu)
    CNN_1_3 = keras.layers.Conv2D(
        filters=h1, kernel_size=5, activation=tf.nn.relu, strides=2)
    pool_1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    x = CNN_1_1(x)
    x_init = CNN_1_1(x_init)
    if use_batch_norm:
        x = batch_1_1(x)
        x_init = batch_1_1(x_init)
    # x = CNN_1_2(x)
    # x_init = CNN_1_2(x_init)
    # if use_batch_norm:
    #     x = batch_1_2(x)
    #     x_init = batch_1_2(x_init)
    x = CNN_1_3(x)
    x_init = CNN_1_3(x_init)
    if use_batch_norm:
        x = batch_1_3(x)
        x_init = batch_1_3(x_init)
    if drop_out_rate > 0:
        drop1 = keras.layers.Dropout(rate=drop_out_rate)
        x = drop1(x)
        x_init = drop1(x_init)
    #x = pool_1(x)
    #x_init = pool_1(x_init)
    if not h2 == 0:
        CNN_2_1 = keras.layers.Conv2D(
            filters=h2, kernel_size=3, activation=tf.nn.relu)
        CNN_2_2 = keras.layers.Conv2D(
            filters=h2, kernel_size=3, activation=tf.nn.relu)
        CNN_2_3 = keras.layers.Conv2D(
            filters=h2, kernel_size=5, activation=tf.nn.relu, strides=2)
        pool_2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        batch_2_1 = keras.layers.BatchNormalization()
        batch_2_2 = keras.layers.BatchNormalization()
        batch_2_3 = keras.layers.BatchNormalization()
        x = CNN_2_1(x)
        x_init = CNN_2_1(x_init)
        if use_batch_norm:
            x = batch_2_1(x)
            x_init = batch_2_1(x_init)
        # x = CNN_2_2(x)
        # x_init = CNN_2_2(x_init)
        # if use_batch_norm:
        #     x = batch_2_2(x)
        #     x_init = batch_2_2(x_init)
        x = CNN_2_3(x)
        x_init = CNN_2_3(x_init)
        if use_batch_norm:
            x = batch_2_3(x)
            x_init = batch_2_3(x_init)
        #x = pool_2(x)
        #x_init = pool_2(x_init)
        if drop_out_rate > 0:
            drop2 = keras.layers.Dropout(rate=drop_out_rate)
            x = drop2(x)
            x_init = drop2(x_init)
    if not h3 == 0:
        CNN_3_1 = keras.layers.Conv2D(
            filters=h3, kernel_size=3, activation=tf.nn.relu)
        CNN_3_2 = keras.layers.Conv2D(
            filters=h3, kernel_size=3, activation=tf.nn.relu)
        CNN_3_3 = keras.layers.Conv2D(
            filters=h3, kernel_size=5, activation=tf.nn.relu, strides=2)
        pool_3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        batch_3_1 = keras.layers.BatchNormalization()
        batch_3_2 = keras.layers.BatchNormalization()
        batch_3_3 = keras.layers.BatchNormalization()
        x = CNN_3_1(x)
        x_init = CNN_3_1(x_init)
        if use_batch_norm:
            x = batch_3_1(x)
            x_init = batch_3_1(x_init)
        x = CNN_3_2(x)
        x_init = CNN_3_2(x_init)
        if use_batch_norm:
            x = batch_3_2(x)
            x_init = batch_3_2(x_init)
        x = CNN_3_3(x)
        x_init = CNN_3_3(x_init)
        if use_batch_norm:
            x = batch_3_3(x)
            x_init = batch_3_3(x_init)
        #x = pool_3(x)
        #x_init = pool_3(x_init)
        if drop_out_rate > 0:
            drop3 = keras.layers.Dropout(rate=drop_out_rate)
            x = drop3(x)
            x_init = drop3(x_init)
    x = keras.layers.Flatten()(x)
    x_init = keras.layers.Flatten()(x_init)
    embedLayer = keras.layers.Dense(embed_size)
    x = embedLayer(x)
    x_init = embedLayer(x_init)
    concat_flat = keras.layers.Concatenate()([x, x_init])
    dense1 = keras.layers.Dense(d1, activation=tf.nn.elu)(concat_flat)
    #dense2 = keras.layers.Dense(d2, activation=tf.nn.elu)(dense1)
    out_u = keras.layers.Dense(2)(dense1)
    out = keras.layers.Add()([center_coords, out_u])
    # Wrap everythin in Keras model
    model = keras.Model(inputs=[img, img_init, center_coords], outputs=out)
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=metrics_distance,
                  metrics=['mean_squared_error', metrics_distance])
    return model


def create_model_trial(img_size,
                 h1=32, h2=64, h3=128,
                 embed_size=128, d1=32,
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
    CNN_1 = keras.layers.Conv2D(
        filters=96, kernel_size=7, activation=tf.nn.relu, strides=2)
    pool_1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)
    x = CNN_1(x)
    x_init = CNN_1(x_init)
    # x = pool_1(x)
    #x_init = pool_1(x_init)
    normalize = keras.layers.Lambda(tf.nn.local_response_normalization)
    x = normalize(x)
    x_init = normalize(x_init)
    CNN_2 = keras.layers.Conv2D(
        filters=256, kernel_size=5, activation=tf.nn.relu, strides=2)
    #p ool_2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)
    x = CNN_2(x)
    x_init = CNN_2(x_init)
    #x = pool_2(x)
    #x_init = pool_2(x_init)
    normalize2 = keras.layers.Lambda(tf.nn.local_response_normalization)
    x = normalize2(x)
    x_init = normalize2(x_init)
    CNN_3 = keras.layers.Conv2D(
        filters=384, kernel_size=3, activation=tf.nn.relu, strides=1)
    CNN_4 = keras.layers.Conv2D(
        filters=256, kernel_size=3, activation=tf.nn.relu, strides=1)
    pool_3 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)
    x = CNN_3(x)
    x_init = CNN_3(x_init)
    x = CNN_4(x)
    x_init = CNN_4(x_init)
   # x = pool_3(x)
   # x_init = pool_3(x_init)
    x = keras.layers.Flatten()(x)
    x_init = keras.layers.Flatten()(x_init)
    d1 = keras.layers.Dense(512)
    #embed = keras.layers.Dense(256)
    x = d1(x)
    x_init = d1(x_init)
    #x = embed(x)
    #x_init = embed(x_init)
    concat_flat = keras.layers.Concatenate()([x, x_init])
    dense1 = keras.layers.Dense(64, activation=tf.nn.elu)(concat_flat)
    out_u = keras.layers.Dense(2)(dense1)
    out = keras.layers.Add()([center_coords, out_u])
    # Wrap everythin in Keras model
    model = keras.Model(inputs=[img, img_init, center_coords], outputs=out)
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=metrics_distance,
                  metrics=['mean_squared_error', metrics_distance])
    return model
