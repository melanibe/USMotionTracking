import tensorflow as tf
from tensorflow import keras
from dataLoader import metrics_distance

'''
Mélanie Bernhardt - ETH Zurich
CLUST Challenge
'''

def create_model_vgg(img_size,
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
    Normalize = keras.layers.Lambda(tf.nn.local_response_normalization)
    Conv1_1 = keras.layers.Conv2D(
        filters=128, kernel_size=3, activation=tf.nn.relu, strides=1)
    Conv1_2 = keras.layers.Conv2D(
        filters=128, kernel_size=3, activation=tf.nn.relu, strides=1)
    Conv1_3 = keras.layers.Conv2D(
        filters=128, kernel_size=3, activation=tf.nn.relu, strides=1)
    Conv2_1 = keras.layers.Conv2D(
        filters=256, kernel_size=3, activation=tf.nn.relu, strides=1)
    Conv2_2 = keras.layers.Conv2D(
        filters=256, kernel_size=3, activation=tf.nn.relu, strides=1)
    Conv2_3 = keras.layers.Conv2D(
        filters=256, kernel_size=3, activation=tf.nn.relu, strides=1)
    Conv3_1 = keras.layers.Conv2D(
        filters=256, kernel_size=3, activation=tf.nn.relu, strides=1)
    Conv3_2 = keras.layers.Conv2D(
        filters=256, kernel_size=3, activation=tf.nn.relu, strides=1)
    Conv3_3 = keras.layers.Conv2D(
        filters=256, kernel_size=3, activation=tf.nn.relu, strides=1)
    MaxPool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    MaxPool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    Embedding = keras.layers.Dense(2048, activation=tf.nn.relu)
    Dense1 = keras.layers.Dense(512, activation=tf.nn.relu)
    Output = keras.layers.Dense(2)
    x = Conv1_1(x)
    x_init = Conv1_1(x_init)
    x = Conv1_2(x)
    x_init = Conv1_2(x_init)
    x = Conv1_3(x)
    x_init = Conv1_3(x_init)
    x = MaxPool1(x)
    x_init = MaxPool1(x_init)
    x = Conv2_1(x)
    x_init = Conv2_1(x_init)
    x = Conv2_2(x)
    x_init = Conv2_2(x_init)
    x = Conv2_3(x)
    x_init = Conv2_3(x_init)
    x = MaxPool2(x)
    x_init = MaxPool2(x_init)
    x = Conv3_1(x)
    x_init = Conv3_1(x_init)
    x = Conv3_2(x)
    x_init = Conv3_2(x_init)
    x = Conv3_3(x)
    x_init = Conv3_3(x_init)
    print(x.shape)
    x = keras.layers.Flatten()(x)
    x_init = keras.layers.Flatten()(x_init)
    x = Embedding(x)
    x_init = Embedding(x_init)
    out = keras.layers.Concatenate()([x, x_init])
    out = Dense1(out)
    out = Output(out)
    out = keras.layers.Add()([center_coords, out])
    # Wrap everythin in Keras model
    model = keras.Model(inputs=[img, img_init, center_coords], outputs=out)
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss=metrics_distance,
                  metrics=['mean_squared_error', metrics_distance])
    return model




def create_model(img_size,
                 h1=32, h2=64, h3=128,
                 embed_size=128, d1=128,
                 drop_out_rate=0.1,
                 use_batch_norm=True):
    """ This functions defines initializes the model.
    """
    center_coords = keras.layers.Input(
        (2,), dtype='float32', name='center_coords')
    big_template = keras.layers.Input((301, 301), dtype='float32', name='big')
    img = keras.layers.Input((img_size, img_size), dtype='float32', name='img')
    img_init = keras.layers.Input(
        shape=(img_size, img_size), dtype='float32', name='img_init')
    big = keras.layers.Reshape((301, 301, 1))(big_template)
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
        filters=h1, kernel_size=5, activation=tf.nn.relu, strides=3)
    pool_1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    x = CNN_1_1(x)
    x_init = CNN_1_1(x_init)
    big = CNN_1_1(big)
    if use_batch_norm:
        x = batch_1_1(x)
        x_init = batch_1_1(x_init)
        big = batch_1_1(big)
   # x = CNN_1_2(x)
    #x_init = CNN_1_2(x_init)
    #if use_batch_norm:
    #     x = batch_1_2(x)
    #     x_init = batch_1_2(x_init)
    x = CNN_1_3(x)
    x_init = CNN_1_3(x_init)
    big = CNN_1_3(big)
    if use_batch_norm:
        x = batch_1_3(x)
        x_init = batch_1_3(x_init)
        big = batch_1_3(big)
    if drop_out_rate > 0:
        drop1 = keras.layers.Dropout(rate=drop_out_rate)
        x = drop1(x)
        x_init = drop1(x_init)
        big = drop1(big)
    #x = pool_1(x)
    #x_init = pool_1(x_init)
    if not h2 == 0:
        CNN_2_1 = keras.layers.Conv2D(
            filters=h2, kernel_size=3, activation=tf.nn.relu, padding='same')
        CNN_2_2 = keras.layers.Conv2D(
            filters=h2, kernel_size=3, activation=tf.nn.relu, padding='same')
        CNN_2_3 = keras.layers.Conv2D(
            filters=h2, kernel_size=5, activation=tf.nn.relu, strides=2)
        pool_2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        batch_2_1 = keras.layers.BatchNormalization()
        batch_2_2 = keras.layers.BatchNormalization()
        batch_2_3 = keras.layers.BatchNormalization()
        x = CNN_2_1(x)
        x_init = CNN_2_1(x_init)
        big = CNN_2_1(big)
        if use_batch_norm:
            x = batch_2_1(x)
            x_init = batch_2_1(x_init)
            big = batch_2_1(big)
        #x = CNN_2_2(x)
        #x_init = CNN_2_2(x_init)
        #if use_batch_norm:
        #     x = batch_2_2(x)
        #     x_init = batch_2_2(x_init)
        x = CNN_2_3(x)
        x_init = CNN_2_3(x_init)
        big = CNN_2_3(big)
        if use_batch_norm:
            x = batch_2_3(x)
            x_init = batch_2_3(x_init)
            big = batch_2_3(big)
        # x = pool_2(x)
        # x_init = pool_2(x_init)
        if drop_out_rate > 0:
            drop2 = keras.layers.Dropout(rate=drop_out_rate)
            x = drop2(x)
            x_init = drop2(x_init)
            big = drop2(big)
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
        big = x = CNN_3_1(big)
        if use_batch_norm:
            x = batch_3_1(x)
            x_init = batch_3_1(x_init)
            big = batch_3_1(big)
        x = CNN_3_2(x)
        x_init = CNN_3_2(x_init)
        big = CNN_3_2(big)
        if use_batch_norm:
            x = batch_3_2(x)
            x_init = batch_3_2(x_init)
            big = batch_3_2(big)
        x = CNN_3_3(x)
        x_init = CNN_3_3(x_init)
        big = CNN_3_3(big)
        if use_batch_norm:
            x = batch_3_3(x)
            x_init = batch_3_3(x_init)
            x_init = batch_3_3(x_init)
        #x = pool_3(x)
        #x_init = pool_3(x_init)
        if drop_out_rate > 0:
            drop3 = keras.layers.Dropout(rate=drop_out_rate)
            x = drop3(x)
            x_init = drop3(x_init)
    print(x.shape)
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


def create_model_axel(img_size,
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
    Normalize = keras.layers.Lambda(tf.nn.local_response_normalization)
    Conv1 = keras.layers.Conv2D(
        filters=48, kernel_size=11, activation=tf.nn.relu, strides=4)
    MaxPool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)
    Conv2 = keras.layers.Conv2D(
        filters=256, kernel_size=5, activation=tf.nn.relu, strides=1, padding='same')
    MaxPool2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)
    Conv3 = keras.layers.Conv2D(
        filters=384, kernel_size=3, activation=tf.nn.relu, strides=1, padding='same')
   # Conv4 = keras.layers.Conv2D(
   #     filters=384, kernel_size=3, activation=tf.nn.relu, strides=1, padding='same')
    Conv5 = keras.layers.Conv2D(
        filters=256, kernel_size=3, activation=tf.nn.relu, strides=1, padding='same')
    Embedding = keras.layers.Dense(512, activation=tf.nn.relu)
    Dense1 = keras.layers.Dense(512, activation=tf.nn.relu)
    Dense2 = keras.layers.Dense(256, activation=tf.nn.relu)
    Output = keras.layers.Dense(2)
    x = Conv1(x)
    x_init = Conv1(x_init)
    x = Normalize(x)
    x_init = Normalize(x_init)
    #x = MaxPool1(x)
    #x_init = MaxPool1(x_init)
    x = Conv2(x)
    x_init = Conv2(x_init)
    x = Normalize(x)
    x_init = Normalize(x_init)
    x = MaxPool2(x)
    x_init = MaxPool2(x_init)    
    x = Conv3(x)
    x_init = Conv3(x_init)
    #x = Conv4(x)
    #x_init = Conv4(x_init)
    x = Conv5(x)
    x_init = Conv5(x_init)
    print(x.shape)
    x = keras.layers.Flatten()(x)
    x_init = keras.layers.Flatten()(x_init)
    x = Embedding(x)
    x_init = Embedding(x_init)
    out = keras.layers.Concatenate()([x, x_init])
    #out = Dense1(out)
    out = Dense2(out)
    out = Output(out)
    out = keras.layers.Add()([center_coords, out])
    # Wrap everythin in Keras model
    model = keras.Model(inputs=[img, img_init, center_coords], outputs=out)
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss=metrics_distance,
                  metrics=['mean_squared_error', metrics_distance])
    return model


def create_model_old(img_size,
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
        filters=32, kernel_size=7, activation=tf.nn.relu, strides=1)
    CNN_1_2 = keras.layers.Conv2D(
        filters=64, kernel_size=5, activation=tf.nn.relu, strides=1)
    #pool_1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    x = CNN_1(x)
    x_init = CNN_1(x_init)
    x = CNN_1_2(x)
    x_init = CNN_1_2(x_init)
    #x = pool_1(x)
    #x_init = pool_1(x_init)
    normalize = keras.layers.Lambda(tf.nn.local_response_normalization)
    x = normalize(x)
    x_init = normalize(x_init)
    CNN_2 = keras.layers.Conv2D(
        filters=128, kernel_size=3, activation=tf.nn.relu, strides=1)
    pool_2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1)
    x = CNN_2(x)
    x_init = CNN_2(x_init)
    x = pool_2(x)
    x_init = pool_2(x_init)
    normalize2 = keras.layers.Lambda(tf.nn.local_response_normalization)
    x = normalize2(x)
    x_init = normalize2(x_init)
    CNN_3 = keras.layers.Conv2D(
        filters=256, kernel_size=3, activation=tf.nn.relu, strides=2)
    CNN_4 = keras.layers.Conv2D(
        filters=256, kernel_size=3, activation=tf.nn.relu, strides=1)
    pool_3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    x = CNN_3(x)
    x_init = CNN_3(x_init)
    x = CNN_4(x)
    x_init = CNN_4(x_init)
    x = pool_3(x)
    x_init = pool_3(x_init)
    x = keras.layers.Flatten()(x)
    x_init = keras.layers.Flatten()(x_init)
    d1 = keras.layers.Dense(512)
    #embed = keras.layers.Dense(256)
    x = d1(x)
    x_init = d1(x_init)
    #x = embed(x)
    #x_init = embed(x_init)
    concat_flat = keras.layers.Concatenate()([x, x_init])
    dense1 = keras.layers.Dense(256, activation=tf.nn.elu)(concat_flat)
    out_u = keras.layers.Dense(2)(dense1)
    out = keras.layers.Add()([center_coords, out_u])
    # Wrap everythin in Keras model
    model = keras.Model(inputs=[img, img_init, center_coords], outputs=out)
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=metrics_distance,
                  metrics=['mean_squared_error', metrics_distance])
    return model
