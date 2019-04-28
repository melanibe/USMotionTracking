from dataLoader import DataLoader, compute_euclidean_distance
from network import create_model
import os
import numpy as np
from custom_KFold import MyKFold
from block_matching_utils import find_template_pixel, global_template_search
from PIL import Image
import pandas as pd
from tensorflow import keras
import logging

np.random.seed(seed=42)
exp_name = 'exp_80'
params_dict = {'dropout_rate': 0.4, 'n_epochs': 20,
               'h3': 0, 'embed_size': 128, 'width': 80}

# ============ DATA AND SAVING DIRS SETUP ========== #
data_dir = os.getenv('DATA_PATH')
exp_dir = os.getenv('EXP_PATH')
checkpoint_dir = os.path.join(exp_dir, exp_name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# ============= LOGGER SETUP ================= #
# create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('my_log')
# create console handler and set level to debug
ch = logging.StreamHandler()
# create formatter
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
log_filename = checkpoint_dir + '/logfile' + '.log'
file_handler = logging.FileHandler(log_filename)
# file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Set the default parameters
if params_dict.get('width') is None:
    params_dict['width'] = 60
if params_dict.get('n_epochs') is None:
    params_dict['n_epochs'] = 15
if params_dict.get('h1') is None:
    params_dict['h1'] = 32
if params_dict.get('h2') is None:
    params_dict['h2'] = 64
if params_dict.get('h3') is None:
    params_dict['h3'] = 0
if params_dict.get('embed_size') is None:
    params_dict['embed_size'] = 64
if params_dict.get('dropout_rate') is None:
    params_dict['dropout_rate'] = 0
if params_dict.get('use_batchnorm') is None:
    params_dict['use_batchnorm'] = True

# ========= PRINT CONFIG TO LOG ======== #
logger.info('Running %s experiment ...' % exp_name)
logger.info('\n Settings for this expriment are: \n')
for key in params_dict.keys():
    logger.info('  {}: {}'.format(key.upper(), params_dict[key]))
logger.info('Saving checkpoint to {}'.format(checkpoint_dir))


# KFold iterator
kf = MyKFold(data_dir, n_splits=5)
fold_iterator = kf.getFolderIterator()
eucl_dist_per_fold = []
pixel_dist_per_fold = []
for traindirs, testdirs in fold_iterator:
    # TRAIN LOCAL PREDICTION MODEL
    # Generators
    logger.info('############ FOLD #############')
    logger.info('Training folders are {}'.format(traindirs))
    training_generator = DataLoader(
        data_dir, traindirs, 32, width_template=params_dict['width'])
    validation_generator = DataLoader(
        data_dir, testdirs, 32, width_template=params_dict['width'], type='val')

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
                        workers=4)
    '''
    try:
        model.load_weights(os.path.join(checkpoint_dir, 'model22.h5'))
    except OSError:
        print('here')
        model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
		                use_multiprocessing=True,
		                epochs=params_dict['n_epochs'])
        model.save_weights(os.path.join(checkpoint_dir, 'model.h5'))
    
    # PREDICT WITH GLOBAL MATCHING + LOCAL MODEL ON TEST SET
    curr_fold_dist = []
    curr_fold_pix = []
    for testfolder in testdirs:
        annotation_dir = os.path.join(data_dir, testfolder, 'Annotation')
        img_dir = os.path.join(data_dir, testfolder, 'Data')
        list_imgs = [os.path.join(img_dir, dI)
                     for dI in os.listdir(img_dir)
                     if (dI.endswith('png')
                         and not dI.startswith('.'))]
        list_label_files = [os.path.join(annotation_dir, dI) for dI
                            in os.listdir(annotation_dir)
                            if (dI.endswith('txt')
                                and not dI.startswith('.'))]
        list_imgs.sort()  # very important to keep the index order
        list_label_files.sort()
        print(list_label_files)
        img_init = np.asarray(Image.open(list_imgs[0]))
        for label_file in list_label_files:
            img_current = np.asarray(Image.open(list_imgs[0]))
            df = pd.read_csv(label_file,
                             header=None,
                             names=['id', 'x', 'y'],
                             sep='\s+')
            c1_init, c2_init = df.loc[df['id'] == 1, ['x', 'y']].values[0, :]
            list_centers = [[c1_init, c2_init]]
            c1, c2 = df.loc[df['id'] == 1, ['x', 'y']].values[0, :]
            xax, yax = find_template_pixel(c1_init, c2_init,
                                           width=params_dict['width'])
            template_init = img_init[np.ravel(yax), np.ravel(
                xax)].reshape(1, len(yax), len(xax))
            n_obs = len(list_imgs)
            print('list_imgs length {}'.format(n_obs))
            for i in range(1, len(list_imgs)):
                if i % 100 == 0:
                    print(i)
                img_prev = img_current
                # modify like in DataLoader
                img_current = np.asarray(Image.open(list_imgs[i]))
                c1, c2, maxNCC = global_template_search(c1,
                                                        c2,
                                                        img_prev,
                                                        img_current,
                                                        width=params_dict['width'])
                xax, yax = find_template_pixel(c1, c2,
                                               width=params_dict['width'])
                template_current = img_current[np.ravel(
                    yax), np.ravel(xax)].reshape(1, len(yax), len(xax))
                current_centers = np.asarray([c1, c2]).reshape(1, 2)
                pred = model.predict(
                    x=[template_current, template_init, current_centers])
                old_c1, old_c2 = c1, c2
                c1, c2 = pred[0, 0], pred[0, 1]
                if np.sqrt((old_c1-c1)**2+(old_c2-c2)**2)>10:
                    print('WARN: weird prediction keep maxNCC pred')
                    c1, c2 = old_c1, old_c2
                list_centers = np.append(list_centers, [c1, c2])
                if i in df.id.values:
                    true = df.loc[df['id'] == i, ['x', 'y']].values[0]
                    diff_x = np.abs(c1-true[0])
                    diff_y = np.abs(c2-true[1])
                    orig_dist = np.sqrt(
                        np.abs(old_c1 - true[0])**2 + np.abs(old_c2 - true[1]))
                    dist = np.sqrt(diff_x**2+diff_y**2)
                    #print('Abs diff in x {}, in y {}'.format(diff_x, diff_y))
                    print('ID {} : dist diff {}'.format(i, dist))
                    print('Init dist before local {}'.format(orig_dist))
                    if dist > 3:
                        print('Bad dist - maxNCC was {}'.format(maxNCC))
            idx = df.id.values.astype(int)
            idx = np.delete(idx, 0)
            list_centers = list_centers.reshape(-1,2)
            df_preds = list_centers[idx-1]
            df_true = df[['x', 'y']].values
            try:
                assert len(idx)==len(df_true)
            except AssertionError:
                print(label_file)
                print(len(idx))
                print(len(df_true))
            df_true = np.delete(df_true, 0, 0)
            absolute_diff = np.mean(np.abs(df_preds-df_true))
            pix_dist = np.mean(
                    np.sqrt((df_preds[:, 0]-df_true[:,0])**2+(df_preds[:,1]-df_true[:,1])**2))
            dist = compute_euclidean_distance(
                kf.resolution_df, testfolder, df_preds, df_true)
            curr_fold_dist.append(dist)
            curr_fold_pix.append(pix_dist)
            logger.info('======== Test Feature {} ======='.format(label_file))
            logger.info('Pixel distance is {}'.format(pix_dist))
            logger.info('Euclidean distance in mm {}'.format(dist))
            logger.info(
                'Mean absolute difference in pixels {}'.format(absolute_diff))
    eucl_dist_per_fold = np.append(eucl_dist_per_fold, np.mean(curr_fold_dist))
    pixel_dist_per_fold = np.append(
        pixel_dist_per_fold, np.mean(curr_fold_pix))
    logger.info('EUCLIDEAN DISTANCE CURRENT FOLD {}'.format(
        eucl_dist_per_fold[-1]))
    logger.info('PIXEL DISTANCE CURRENT FOLD {}'.format(
        pixel_dist_per_fold[-1]))
logger.info('================= END RESULTS =================')
logger.info('Mean euclidean distance in mm {} (std {})'
            .format(np.mean(eucl_dist_per_fold),
                    np.std(eucl_dist_per_fold)))
