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
import tensorflow as tf
import parmap
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
'''
Mélanie Bernhardt - ETH Zurich
CLUST Challenge
'''


def get_next_center(c1_prev, c2_prev, img_prev, img_current,
                    params_dict, model, template_init, logger=None, est_c1=None, est_c2=None, c1_hist=None, c2_hist=None):
    c1, c2, maxNCC = NCC_best_template_search(c1_prev,
                                              c2_prev,
                                              img_prev,
                                              img_current,
                                              width=params_dict['width'],
                                              search_w=params_dict['search_w'])
    xax, yax = find_template_pixel(c1, c2,
                                   params_dict['width'], img_current.shape[1], img_current.shape[0])
    template_current = img_current[np.ravel(
        yax), np.ravel(xax)].reshape(1, len(yax), len(xax))
    current_centers = np.asarray([c1, c2]).reshape(1, 2)
    pred = model.predict(
        x=[template_current, template_init, current_centers])
    old_c1, old_c2 = c1, c2
    c1, c2 = pred[0, 0], pred[0, 1]
    if est_c1 is not None:
        c1_temp = est_c1.predict(c1_hist.reshape(1, -1))
        c2_temp = est_c2.predict(c2_hist.reshape(1, -1))
        if np.sqrt((c1_temp-c1)**2+(c2_temp-c2)**2) > 5:
            if logger is None:
                print('WARN: using temporal pred')
            else:
                logger.info('WARN: using temporal pred')
                logger.info('temp {}, {}'.format(c1_temp, c2_temp))
                logger.info('net {}, {}'.format(c1, c2))
            c1, c2 = np.mean([c1_temp, c1]), np.mean([c2_temp, c2])
    if maxNCC < 0.85:
        logger.info('WARN WARN MAX NCC {}'.format(maxNCC))
        c1_save, c2_save, other_maxNCC = NCC_best_template_search(c1_prev,
                                                c2_prev,
                                                img_prev,
                                                img_current,
                                                width=params_dict['width'],
                                                search_w=20)
        if other_maxNCC > 0.90:
            c1, c2 = np.mean([c1_temp, c1_save]), np.mean([c2_temp, c2_save])
    """
    if np.sqrt((c1_prev-c1)**2+(c2_prev-c2)**2) > 10:
        if np.sqrt((old_c1-c1_prev)**2+(old_c2-c2_prev)**2) < np.sqrt((old_c1-c1)**2+(old_c2-c2)**2):
            if np.sqrt((old_c1-c1_prev)**2+(old_c2-c2_prev)**2) < 5:
                if logger is None:
                    print('WARN: weird prediction mean maxNCC old_pred')
                    print('previous {},{}'.format(c1_prev, c2_prev))
                    print('NCC {},{}'.format(old_c1, old_c2))
                    print('proposed by net {},{}'.format(c1, c2))
                    print('kept {},{}'.format(
                        (old_c1+c1_prev)/2, (old_c2+c2_prev)/2))
                else:
                    logger.info(
                        'WARN: weird prediction mean both maxNCC old_pred')
                    logger.info('previous {},{}'.format(c1_prev, c2_prev))
                    logger.info('NCC {},{}'.format(old_c1, old_c2))
                    logger.info('proposed by net {},{}'.format(c1, c2))
                    logger.info('kept {},{}'.format(
                        (old_c1+c1_prev)/2, (old_c2+c2_prev)/2))
                c1, c2 = (old_c1+c1_prev)/2, (old_c2+c2_prev)/2
        else:
            if logger is None:
                print('WARN: VERY VERY weird prediction keep old_pred')
                print('previous {},{}'.format(c1_prev, c2_prev))
                print('NCC {},{}'.format(old_c1, old_c2))
                print('proposed by net {},{}'.format(c1, c2))
                print('kept {},{}'.format(c1_prev, c2_prev))
            else:
                logger.info(
                    'WARN: VERY VERY weird prediction keep old_pred')
                logger.info('previous {},{}'.format(c1_prev, c2_prev))
                logger.info('NCC {},{}'.format(old_c1, old_c2))
                logger.info('proposed by net {},{}'.format(c1, c2))
                logger.info('kept {},{}'.format(
                    c1_prev, c2_prev))
                c1, c2 = c1_prev, c2_prev
        # else:
        #     if logger is None:
        #         print('WARN: weird prediction mean both maxNCC current_pred')
        #     else:
        #         logger.info(
        #             'WARN: weird prediction mean both maxNCC current_pred')
        #         logger.info('previous {},{}'.format(c1_prev, c2_prev))
        #         logger.info('NCC {},{}'.format(old_c1, old_c2))
        #         logger.info('proposed by net {},{}'.format(c1, c2))
        #         logger.info('kept {},{}'.format((old_c1+c1)/2, (old_c2+c2)/2))
        #     c1, c2 = (old_c1+c1)/2, (old_c2+c2)/2
        """
    return c1, c2, old_c1, old_c2, maxNCC


def run_global_cv(fold_iterator, logger, params_dict, upsample=True):
    eucl_dist_per_fold = []
    pixel_dist_per_fold = []
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
            type='val', upsample=upsample)

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
            model.load_weights(os.path.join(checkpoint_dir, 'model.h5'))
        except OSError:
            print('here')
            model.fit_generator(generator=training_generator,
                                validation_data=validation_generator,
                                use_multiprocessing=True,
                                epochs=params_dict['n_epochs'],
                                workers=4, max_queue_size=20)
            model.save_weights(os.path.join(checkpoint_dir, 'model.h5'))
        for folder in traindirs:
            res_x, res_y = training_generator.resolution_df.loc[
                training_generator.resolution_df['scan']
                == folder, ['res_x', 'res_y']].values[0]
            img_dir = os.path.join(data_dir, folder, 'Data')
            annotation_dir = os.path.join(data_dir, folder, 'Annotation')
            list_label_files = [os.path.join(annotation_dir, dI) for dI
                                in os.listdir(annotation_dir)
                                if (dI.endswith('txt')
                                    and not dI.startswith('.'))]
            try:
                img_init = np.asarray(Image.open(
                    os.path.join(img_dir, "{:04d}.png".format(1))))
            except FileNotFoundError:
                img_init = np.asarray(Image.open(
                    os.path.join(img_dir, "{:05d}.png".format(1))))
            # img_init = prepare_input_img(img_init, res_x, res_y, upsample)
            list_imgs = [os.path.join(img_dir, dI)
                         for dI in os.listdir(img_dir)
                         if (dI.endswith('png')
                             and not dI.startswith('.'))]
            n_obs = len(list_imgs)
            X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = [
            ], [], [], [], [], [], [], [], [], [], []
            Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10 = [
            ], [], [], [], [], [], [], [], [], [], []
            for label in list_label_files:
                print(label)
                df = pd.read_csv(os.path.join(annotation_dir, label),
                                 header=None,
                                 names=['id', 'x', 'y'],
                                 sep='\s+')
                c1_interpolate = np.interp(
                    np.arange(1, n_obs+1), df.id.values, df.x.values)
                c2_interpolate = np.interp(
                    np.arange(1, n_obs+1), df.id.values, df.y.values)
                n = len(c1_interpolate)
                X0 = np.append(X0, c1_interpolate)
                X1 = np.append(X1, c1_interpolate[1:n])
                X2 = np.append(X2, c1_interpolate[2:n])
                X3 = np.append(X3, c1_interpolate[3:n])
                X4 = np.append(X4, c1_interpolate[4:n])
                X5 = np.append(X5, c1_interpolate[5:n])
                """
                X6 = np.append(X6, c1_interpolate[6:n])
                X7 = np.append(X7, c1_interpolate[7:n])
                X8 = np.append(X8, c1_interpolate[8:n])
                X9 = np.append(X9, c1_interpolate[9:n])
                X10 = np.append(X10, c1_interpolate[10:n])
                """
                Y0 = np.append(Y0, c2_interpolate)
                Y1 = np.append(Y1, c2_interpolate[1:n])
                Y2 = np.append(Y2, c2_interpolate[2:n])
                Y3 = np.append(Y3, c2_interpolate[3:n])
                Y4 = np.append(Y4, c2_interpolate[4:n])
                Y5 = np.append(Y5, c2_interpolate[5:n])
                """
                Y6 = np.append(Y6, c2_interpolate[6:n])
                Y7 = np.append(Y7, c2_interpolate[7:n])
                Y8 = np.append(Y8, c2_interpolate[8:n])
                Y9 = np.append(Y9, c2_interpolate[9:n])
                Y10 = np.append(Y10, c2_interpolate[10:n])
                """
        l = len(X5)
        fullX = np.transpose(np.vstack(
            [X0[0:l], X1[0:l], X2[0:l], X3[0:l], X4[0:l]]))
        fullY = np.transpose(np.vstack(
            [Y0[0:l], Y1[0:l], Y2[0:l], Y3[0:l], Y4[0:l]]))
        c1_label = X5
        c2_label = Y5
        est_c1 = RidgeCV()
        est_c2 = RidgeCV()
        #est_small_c1 = RidgeCV()
        #est_small_c2 = RidgeCV()
        logger.info('c1')
        scores_c1 = cross_validate(est_c1, fullX, c1_label, cv=5, scoring=(
            'r2', 'neg_mean_squared_error'))
        logger.info(scores_c1['test_neg_mean_squared_error'])
        logger.info('c2')
        scores_c2 = cross_validate(est_c2, fullY, c2_label, cv=5, scoring=(
            'r2', 'neg_mean_squared_error'))
        logger.info(scores_c2['test_neg_mean_squared_error'])
        # small_scores_c1 = cross_validate(est_small_c1, smallFullX, c1_label, cv=5, scoring=(
        #     'r2', 'neg_mean_squared_error'))
        # logger.info('Small c1')
        # logger.info(small_scores_c1['test_neg_mean_squared_error'])
        # small_scores_c2 = cross_validate(est_small_c2, smallFullY, c2_label, cv=5, scoring=(
        #     'r2', 'neg_mean_squared_error'))
        # logger.info('Small c2')
        #logger.info(small_scores_c2['test_neg_mean_squared_error'])
        est_c1.fit(fullX, c1_label)
        est_c2.fit(fullY, c2_label)
        # PREDICT WITH GLOBAL MATCHING + LOCAL MODEL ON TEST SET
        curr_fold_dist = []
        curr_fold_pix = []
        for k, testfolder in enumerate(testdirs):
            if k==0:
                continue
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
            img_init = prepare_input_img(img_init, res_x, res_y, upsample)

            for j, label_file in enumerate(list_label_files):
                print(label_file)
                img_current = img_init
                df = pd.read_csv(os.path.join(annotation_dir, label_file),
                                 header=None,
                                 names=['id', 'x', 'y'],
                                 sep='\s+')
                if upsample:
                    df['x_newres'] = df['x']*res_x/0.4
                    df['y_newres'] = df['y']*res_y/0.4
                else:
                    df['x_newres'] = df['x']
                    df['y_newres'] = df['y']
                c1_init, c2_init = df.loc[df['id'] == 1, [
                    'x_newres', 'y_newres']].values[0, :]
                a, b = np.nonzero(img_init[:, 20:(len(img_init)-20)])
                if upsample:
                    list_centers = [[c1_init*0.4/res_x, c2_init*0.4/res_y]]
                else:
                    list_centers = [[c1_init, c2_init]]
                xax, yax = find_template_pixel(c1_init, c2_init,
                                               params_dict['width'], img_init.shape[1], img_init.shape[0])
                template_init = img_init[np.ravel(yax), np.ravel(
                    xax)].reshape(1, len(yax), len(xax))
                c1, c2 = c1_init, c2_init
                for i in range(2, len(list_imgs)+1):
                    if i % 100 == 0:
                        print(i)
                    img_prev = img_current
                    try:
                        img_current = np.asarray(Image.open(
                            os.path.join(img_dir, "{:04d}.png".format(i))))
                    except FileNotFoundError:
                        img_current = np.asarray(Image.open(
                            os.path.join(img_dir, "{:05d}.png".format(i))))
                    img_current = prepare_input_img(
                        img_current, res_x, res_y, upsample)
                    if i > 5:
                        tmp = list_centers[-10:].reshape(-1, 2)
                        assert tmp.shape[0] == 5
                        c1, c2, old_c1, old_c2, maxNCC = get_next_center(
                            c1, c2, img_prev, img_current, params_dict, model, template_init, logger, est_c1, est_c2, tmp[:, 0], tmp[:, 1])
                    else:
                        c1, c2, old_c1, old_c2, maxNCC = get_next_center(
                            c1, c2, img_prev, img_current, params_dict, model, template_init, logger)
                    # project back in init coords
                    if upsample:
                        c1_orig_coords = c1*0.4/res_x
                        c2_orig_coords = c2*0.4/res_y
                    else:
                        c1_orig_coords = c1
                        c2_orig_coords = c2
                    list_centers = np.append(
                        list_centers, [c1_orig_coords, c2_orig_coords])
                    if i in df.id.values:
                        true = df.loc[df['id'] == i, ['x', 'y']].values[0]
                        diff_x = np.abs(c1_orig_coords-true[0])
                        diff_y = np.abs(c2_orig_coords-true[1])
                        if upsample:
                            dist = np.sqrt(diff_x**2+diff_y**2)
                            logger.info('ID {} : euclidean dist diff {}'
                                        .format(i, dist*0.4))
                        else:
                            dist = np.sqrt((res_x*diff_x)**2+(diff_y*res_y)**2)
                            logger.info('ID {} : euclidean dist diff {}'
                                        .format(i, dist))
                        if dist > 10:
                            logger.info(
                                'Bad dist - maxNCC was {}'.format(maxNCC))
                            logger.info('True {},{}'.format(true[0], true[1]))
                            logger.info('Pred {},{}'.format(
                                c1_orig_coords, c2_orig_coords))
                            if upsample:
                                logger.info('NCC {},{}'.format(
                                    old_c1*0.4/res_x, old_c2*0.4/res_y))
                            else:
                                logger.info('NCC {},{}'.format(
                                    old_c1, old_c2))
                idx = df.id.values.astype(int)
                list_centers = list_centers.reshape(-1, 2)
                df_preds = list_centers[idx-1]
                df_true = df[['x', 'y']].values
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
                pred_df['idx'] = range(1, len(list_centers)+1)
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
                       model, params_dict, upsample):
    annotation_dir = os.path.join(data_dir, testfolder, 'Annotation')
    img_dir = os.path.join(data_dir, testfolder, 'Data')

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
        img_init = prepare_input_img(img_init, res_x, res_y, upsample)
    list_preds_df = parmap.starmap(predict_feature,
                                   list_label_files,
                                   img_init, img_dir,
                                   res_x, res_y, model,
                                   annotation_dir, params_dict, checkpoint_dir, upsample)
    return list_preds_df


def predict_feature(label_file, img_init,
                    img_dir, res_x, res_y, model, annotation_dir,
                    params_dict, checkpoint_dir, upsample,
                    limit=None, est_c1=None, est_c2=None):
    if limit is None:
        list_imgs = [os.path.join(img_dir, dI)
                     for dI in os.listdir(img_dir)
                     if (dI.endswith('png')
                         and not dI.startswith('.'))]
        n_obs = len(list_imgs)
    else:
        n_obs = limit
    df = pd.read_csv(os.path.join(annotation_dir, label_file),
                     header=None,
                     names=['id', 'x', 'y'],
                     sep='\s+')
    c1_init, c2_init = df.loc[df['id'] == 1, [
        'x', 'y']].values[0, :]
    print(c1_init, c2_init)
    if upsample:
        xax, yax = find_template_pixel(c1_init*res_x/0.4, c2_init*res_y/0.4,
                                       width=params_dict['width'], max_x=img_init.shape[1], max_y=img_init.shape[0])
    else:
        xax, yax = find_template_pixel(c1_init, c2_init,
                                       width=params_dict['width'], max_x=img_init.shape[1], max_y=img_init.shape[0])
    template_init = img_init[np.ravel(yax), np.ravel(
        xax)].reshape(1, len(yax), len(xax))
    img_current = img_init
    list_centers = [[c1_init, c2_init]]
    if upsample:
        c1 = c1_init*res_x/0.4
        c2 = c2_init*res_y/0.4
    else:
        c1 = c1_init
        c2 = c2_init
    for i in range(2, n_obs):
        if i % 50 == 0:
            print(i)
        img_prev = img_current
        try:
            img_current = np.asarray(Image.open(
                os.path.join(img_dir, "{:04d}.png".format(i))))
        except FileNotFoundError:
            img_current = np.asarray(Image.open(
                os.path.join(img_dir, "{:05d}.png".format(i))))
        img_current = prepare_input_img(img_current, res_x, res_y, upsample)
        if ((est_c1 is None) or (i <= 10)):  # during training
            c1, c2, old_c1, old_c2, maxNCC = get_next_center(
                c1, c2, img_prev, img_current, params_dict, model, template_init)
        else:  # at test time use the trained temporal estimator
            tmp = list_centers[-20:].reshape(-1, 2)
            c1, c2, old_c1, old_c2, maxNCC = get_next_center(
                c1, c2, img_prev, img_current, params_dict, model, template_init,
                est_c1, est_c2, tmp[:, 0], tmp[:, 1])
        # project back in init coords
        if upsample:
            c1_orig_coords, c2_orig_coords = c1*0.4/res_x, c2*0.4/res_y
        else:
            c1_orig_coords, c2_orig_coords = c1, c2
        list_centers = np.append(
            list_centers, [c1_orig_coords, c2_orig_coords])
        list_centers = list_centers.reshape(-1, 2)
    pred_df = pd.DataFrame()
    pred_df['id'] = np.arange(1, len(list_centers)+1)
    pred_df['c1'] = list_centers[:, 0]
    pred_df['c2'] = list_centers[:, 1]
    if limit is None:
        pred_df.to_csv(os.path.join(checkpoint_dir, '{}'.format(
            label_file)), header=False, index=False)
    print('{} DONE'.format(label_file))
    return pred_df


if __name__ == '__main__':
    np.random.seed(seed=42)
    exp_name = '2layers_noup_se1_temporal5_epochs15_saveit'
    params_dict = {'dropout_rate': 0.5, 'n_epochs': 15,
                   'h3': 0, 'embed_size': 256, 'width': 60, 'search_w': 1}

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
    run_global_cv(fold_iterator, logger, params_dict, upsample=False)
