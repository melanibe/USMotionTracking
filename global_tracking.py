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

'''
Mélanie Bernhardt - ETH Zurich
CLUST Challenge
'''

def get_next_center(c1_prev, c2_prev, img_prev, img_current,
                    params_dict, model, template_init, logger=None):
    c1, c2, maxNCC = NCC_best_template_search(c1_prev,
                                            c2_prev,
                                            img_prev,
                                            img_current,
                                            width=params_dict['width'],
                                            search_w=params_dict['search_w'])
    xax, yax = find_template_pixel(c1, c2,
                                   width=params_dict['width'])
    template_current = img_current[np.ravel(
        yax), np.ravel(xax)].reshape(1, len(yax), len(xax))
    current_centers = np.asarray([c1, c2]).reshape(1, 2)
    pred = model.predict(
        x=[template_current, template_init, current_centers])
    old_c1, old_c2 = c1, c2
    c1, c2 = pred[0, 0], pred[0, 1]
    if np.sqrt((old_c1-c1)**2+(old_c2-c2)**2) > 15:
        print(old_c1, old_c2)
        print(c1_prev, c2_prev)
        print(c1, c2)
        if np.sqrt((c1_prev-c1)**2+(c1_prev-c2)**2) > 10:
            if logger is None:
                print('WARN: VERY weird prediction mean maxNCC old_pred')
            else:
                logger.info('WARN: VERY weird prediction mean both maxNCC old_pred')
            c1, c2 = (old_c1+c1_prev)/2, (old_c2+c2_prev)/2
        else:
            if logger is None:
                print('WARN: weird prediction mean both maxNCC current_pred')
            else:
                logger.info('WARN: weird prediction mean both maxNCC current_pred')
            c1, c2 = (old_c1+c1)/2, (old_c2+c2)/2
    return c1, c2, maxNCC


def run_global_cv(fold_iterator, logger, params_dict):
    eucl_dist_per_fold = []
    pixel_dist_per_fold = []
    for traindirs, testdirs in fold_iterator:
        # TRAIN LOCAL PREDICTION MODEL
        # Generators
        logger.info('############ FOLD #############')
        logger.info('Training folders are {}'.format(traindirs))
        training_generator = DataLoader(
            data_dir, traindirs, 32,
            width_template=params_dict['width'])
        validation_generator = DataLoader(
            data_dir, testdirs, 32,
            width_template=params_dict['width'],
            type='val')

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
                                workers=6)
            model.save_weights(os.path.join(checkpoint_dir, 'model.h5'))

        # PREDICT WITH GLOBAL MATCHING + LOCAL MODEL ON TEST SET
        curr_fold_dist = []
        curr_fold_pix = []
        for k, testfolder in enumerate(testdirs):
            res_x, res_y = training_generator.resolution_df.loc[
                training_generator.resolution_df['scan']
                == testfolder, ['res_x', 'res_y']].values[0]

            annotation_dir = os.path.join(data_dir, testfolder, 'Annotation')
            img_dir = os.path.join(data_dir, testfolder, 'Data')
            list_imgs = [os.path.join(img_dir, dI)
                         for dI in os.listdir(img_dir)
                         if (dI.endswith('png')
                             and not dI.startswith('.'))]

            list_label_files = [dI for dI
                                in os.listdir(annotation_dir)
                                if (dI.endswith('txt')
                                    and not dI.startswith('.'))]
            print(list_label_files)
            try:
                img_init = np.asarray(Image.open(
                    os.path.join(img_dir, "{:04d}.png".format(1))))
            except FileNotFoundError:
                img_init = np.asarray(Image.open(
                    os.path.join(img_dir, "{:05d}.png".format(1))))
            img_init = prepare_input_img(img_init, res_x, res_y)
            for j, label_file in enumerate(list_label_files):
                img_current = img_init
                df = pd.read_csv(os.path.join(annotation_dir, label_file),
                                 header=None,
                                 names=['id', 'x', 'y'],
                                 sep='\s+')
                df['x_newres'] = df['x']*res_x/0.27
                df['y_newres'] = df['y']*res_y/0.27
                c1_init, c2_init = df.loc[df['id'] == 1, [
                    'x_newres', 'y_newres']].values[0, :]
                list_centers = [[c1_init*0.27/res_x, c2_init*0.27/res_y]]
                xax, yax = find_template_pixel(c1_init, c2_init,
                                               width=params_dict['width'])
                template_init = img_init[np.ravel(yax), np.ravel(
                    xax)].reshape(1, len(yax), len(xax))
                c1, c2 = c1_init, c2_init
                for i in range(2, len(list_imgs)):
                    if i % 100 == 0:
                        print(i)
                    img_prev = img_current
                    try:
                        img_current = np.asarray(Image.open(
                            os.path.join(img_dir, "{:04d}.png".format(i))))
                    except FileNotFoundError:
                        img_current = np.asarray(Image.open(
                            os.path.join(img_dir, "{:05d}.png".format(i))))
                    img_current = prepare_input_img(img_current, res_x, res_y)
                    c1, c2, maxNCC = get_next_center(
                        c1, c2, img_prev, img_current, params_dict, model, template_init, logger)
                    # project back in init coords
                    c1_orig_coords = c1*0.27/res_x
                    c2_orig_coords = c2*0.27/res_y
                    list_centers = np.append(
                        list_centers, [c1_orig_coords, c2_orig_coords])
                    if i in df.id.values:
                        true = df.loc[df['id'] == i, ['x', 'y']].values[0]
                        diff_x = np.abs(c1_orig_coords-true[0])
                        diff_y = np.abs(c2_orig_coords-true[1])
                        dist = np.sqrt(diff_x**2+diff_y**2)
                        print('ID {} : euclidean dist diff {}'
                              .format(i, dist*0.27))
                        print('ID {} : pixel dist diff {}'.format(i, dist))
                        if dist > 3:
                            print('Bad dist - maxNCC was {}'.format(maxNCC))
                idx = df.id.values.astype(int)
                idx = np.delete(idx, 0)
                list_centers = list_centers.reshape(-1, 2)
                df_preds = list_centers[idx-1]
                df_true = df[['x', 'y']].values
                try:
                    assert len(idx) == len(df_true)
                except AssertionError:
                    print(label_file)
                    print(len(idx))
                    print(len(df_true))
                df_true = np.delete(df_true, 0, 0)
                absolute_diff = np.mean(np.abs(df_preds-df_true))
                pix_dist = np.mean(
                    np.sqrt((df_preds[:, 0]-df_true[:, 0]) ** 2 +
                            (df_preds[:, 1]-df_true[:, 1]) ** 2))
                dist = compute_euclidean_distance(df_preds, df_true)
                curr_fold_dist.append(dist)
                curr_fold_pix.append(pix_dist)
                logger.info(
                    '======== Test Feature {} ======='.format(label_file))
                logger.info('Pixel distance is {}'.format(pix_dist))
                logger.info('Euclidean distance in mm {}'.format(dist))
                logger.info(
                    'Mean absolute difference in pixels {}'
                    .format(absolute_diff))
                pred_df = pd.DataFrame()
                pred_df['idx'] = range(len(list_centers))
                pred_df['c1'] = list_centers[:, 0]
                pred_df['c2'] = list_centers[:, 1]
                pred_df.to_csv(os.path.join(checkpoint_dir, '{}.txt'.format(
                    label_file)), header=False, index=False)
        eucl_dist_per_fold = np.append(
            eucl_dist_per_fold, np.mean(curr_fold_dist))
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


def predict_testfolder(testfolder, data_dir, res_x, res_y,
                       model, params_dict):
    annotation_dir = os.path.join(data_dir, testfolder, 'Annotation')
    img_dir = os.path.join(data_dir, testfolder, 'Data')
    list_imgs = [os.path.join(img_dir, dI)
                 for dI in os.listdir(img_dir)
                 if (dI.endswith('png')
                     and not dI.startswith('.'))]

    list_label_files = [dI for dI
                        in os.listdir(annotation_dir)
                        if (dI.endswith('txt')
                            and not dI.startswith('.'))]
    print(list_label_files)
    try:
        img_init = np.asarray(Image.open(
            os.path.join(img_dir, "{:04d}.png".format(1))))
    except FileNotFoundError:
        img_init = np.asarray(Image.open(
            os.path.join(img_dir, "{:05d}.png".format(1))))
        img_init = prepare_input_img(img_init, res_x, res_y)
    for j, label_file in enumerate(list_label_files):
        df = pd.read_csv(os.path.join(annotation_dir, label_file),
                         header=None,
                         names=['id', 'x', 'y'],
                         sep='\s+')
        c1_init, c2_init = df.loc[df['id'] == 1, [
            'x', 'y']].values[0, :]
        xax, yax = find_template_pixel(c1_init*res_x/0.27, c2_init*res_y*0.27,
                                       width=params_dict['width'])
        template_init = img_init[np.ravel(yax), np.ravel(
            xax)].reshape(1, len(yax), len(xax))
        pred_df = predict_feature(c1_init, c2_init, img_init, len(
            list_imgs), img_dir, res_x, res_y, model, template_init, params_dict)
        pred_df.to_csv(os.path.join(checkpoint_dir, '{}'.format(
            label_file)), header=False, index=False)
        print('{} DONE, {}/{}'.format(label_file, j, len(list_label_files)))


def predict_feature(c1_init, c2_init, img_init, n_obs,
                    img_dir, res_x, res_y, model, template_init, params_dict):
    img_current = img_init
    list_centers = [[c1_init, c2_init]]
    c1 = c1_init*res_x/0.27
    c2 = c2_init*res_y/0.27
    for i in range(2, n_obs):
        if i % 100 == 0:
            print(i)
        img_prev = img_current
        try:
            img_current = np.asarray(Image.open(
                os.path.join(img_dir, "{:04d}.png".format(i))))
        except FileNotFoundError:
            img_current = np.asarray(Image.open(
                os.path.join(img_dir, "{:05d}.png".format(i))))
        img_current = prepare_input_img(img_current, res_x, res_y)
        c1, c2, maxNCC = get_next_center(
            c1, c2, img_prev, img_current, params_dict, model, template_init)
        # project back in init coords
        c1_orig_coords, c2_orig_coords = c1*0.27/res_x, c2*0.27/res_y
        list_centers = np.append(
            list_centers, [c1_orig_coords, c2_orig_coords])
        list_centers = list_centers.reshape(-1, 2)
    pred_df = pd.DataFrame()
    pred_df['id'] = np.arange(1, n_obs)
    pred_df['c1'] = list_centers[:, 0]
    pred_df['c2'] = list_centers[:, 1]
    return pred_df


if __name__ == '__main__':
    np.random.seed(seed=42)
    exp_name = 'new_exp_60_10_128_60'
    params_dict = {'dropout_rate': 0.5, 'n_epochs': 25,
                   'h3': 0, 'embed_size': 128, 'width': 60, 'search_w': 60}

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

    run_global_cv(fold_iterator, logger, params_dict)
