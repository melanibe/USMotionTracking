from dataLoader import DataLoader, compute_euclidean_distance
import numpy as np
import os
from custom_KFold import MyKFold
from network import create_model
import pandas as pd

'''
Mélanie Bernhardt - ETH Zurich
CLUST Challenge
'''
"""
def run_CV(params_dict, data_dir, n_splits=5):
    width = params_dict.get('width') if params_dict.get(
        'width') is not None else 60
    n_epochs = params_dict.get('n_epochs') if params_dict.get(
        'n_epochs') is not None else 15
    h1 = params_dict.get('h1') if params_dict.get('h1') is not None else 32
    h2 = params_dict.get('h2') if params_dict.get('h2') is not None else 64
    h3 = params_dict.get('h3') if params_dict.get('h3') is not None else 128
    embed_size = params_dict.get('embed_size') if params_dict.get(
        'embed_size') is not None else 128
    dropout_rate = params_dict.get('dropout_rate') if params_dict.get(
        'dropout_rate') is not None else 0
    use_batchnorm = params_dict.get('batch_norm') if params_dict.get(
        'batch_norm') is not None else True
    # KFold iterator
    kf = MyKFold(data_dir, n_splits=n_splits)
    fold_iterator = kf.getFolderIterator()
    eucl_dist_dict = {}
    eucl_dist_per_fold = []
    mse_per_fold = []
    for traindirs, testdirs in fold_iterator:
        # Generators
        training_generator = DataLoader(
            data_dir, traindirs, 32, width_template=width)
        validation_generator = DataLoader(
            data_dir, testdirs, 32, width_template=width, type='val')
        # Design model
        model = create_model(width+1, h1, h2, h3,
                             embed_size=embed_size,
                             drop_out_rate=dropout_rate,
                             use_batch_norm=use_batchnorm)
        # Train model on dataset
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            epochs=n_epochs,
                            workers=4)
        list_dist_curr_fold = []
        mse_curr_fold = []
        for testfolder in testdirs:
            eucl_dist = 0
            batch_counter = 0
            gen = DataLoader(data_dir, [testfolder],
                             1, width_template=width, type='val')
            mse_curr_fold.append(model.evaluate_generator(gen))
            n = gen.__len__()
            print(n)
            for data, labels in gen:
                if batch_counter < n:
                    batch_counter += 1
                    preds = model.predict(x=data)
                    eucl_dist += compute_euclidean_distance(
                        gen.resolution_df, testfolder, preds, labels)
                else:
                    break
            print(eucl_dist/batch_counter)
            eucl_dist_dict[testfolder] = eucl_dist/batch_counter
            list_dist_curr_fold.append(eucl_dist/batch_counter)
        eucl_dist_per_fold.append(np.mean(list_dist_curr_fold))
        mse_per_fold.append(np.mean(mse_curr_fold))
    print(np.mean(eucl_dist_per_fold))
    return eucl_dist_per_fold, eucl_dist_dict
    # labels = model.predict(validation_generator)

"""

from dataLoader import DataLoader, compute_euclidean_distance, prepare_input_img
from network import create_model
import os
import numpy as np
from custom_KFold import MyKFold
from block_matching_utils import find_template_pixel, NCC_best_template_search
from PIL import Image
import pandas as pd
from tensorflow import keras
from utils import get_logger, get_default_params
import skimage
import tensorflow as tf

'''
Mélanie Bernhardt - ETH Zurich
CLUST Challenge
'''

def run_cv(fold_iterator, logger, params_dict, upsample=True):
    for traindirs, testdirs in fold_iterator:
        # TRAIN LOCAL PREDICTION MODEL
        # Generators
        logger.info('############ FOLD #############')
        logger.info('Training folders are {}'.format(traindirs))
        training_generator = DataLoader(
            data_dir, traindirs, 32,
            width_template=params_dict['width'], upsample=upsample)
        validation_generator = DataLoader(
            data_dir, testdirs, 32,
            width_template=params_dict['width'],
            type='val',upsample=upsample)

        # Design model
        model = create_model(params_dict['width']+1,
                             params_dict['h1'],
                             params_dict['h2'],
                             params_dict['h3'],
                             embed_size=params_dict['embed_size'],
                             drop_out_rate=params_dict['dropout_rate'],
                             use_batch_norm=params_dict['use_batchnorm'])
        # Train model on training dataset
        '''
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            epochs=params_dict['n_epochs'],
                            workers=6)
        '''
        try:
            model.load_weights(os.path.join(checkpoint_dir, 'model22.h5'))
        except OSError:
            print('here')
            model.fit_generator(generator=training_generator,
                                validation_data=validation_generator,
                                use_multiprocessing=True,
                                epochs=params_dict['n_epochs'],
                                workers=4, max_queue_size=20)
            model.save_weights(os.path.join(checkpoint_dir, 'model.h5'))
        metrics = model.evaluate_generator(generator=validation_generator, workers=4, max_queue_size=20)
        logger.info(metrics)

if __name__ == '__main__':
    np.random.seed(seed=42)
    exp_name = 'new_cv_alone_2layers_25_up'
    params_dict = {'dropout_rate': 0.5, 'n_epochs': 10,
                   'h3': 0, 'embed_size': 256, 'width': 100, 'search_w': 50}

    # ============ DATA AND SAVING DIRS SETUP ========== #
    data_dir = os.getenv('DATA_PATH')
    exp_dir = os.getenv('EXP_PATH')
    checkpoint_dir = os.path.join(exp_dir, exp_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # ============= LOGGER SETUP ================= #
    # create logger
    logger = get_logger(checkpoint_dir)

    # Set the default parameters
    params_dict = get_default_params(params_dict)

    # ========= PRINT CONFIG TO LOG ======== #
    logger.info('Running %s experiment ...' % exp_name)
    logger.info('\n Settings for this expriment are: \n')
    for key in params_dict.keys():
        logger.info('  {}: {}'.format(key.upper(), params_dict[key]))
    logger.info('Saving checkpoint to {}'.format(checkpoint_dir))

    # KFold iterator
    kf = MyKFold(data_dir, n_splits=5)
    fold_iterator = kf.getFolderIterator()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(sess)
    run_cv(fold_iterator, logger, params_dict,upsample=True)

