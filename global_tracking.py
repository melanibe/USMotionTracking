from dataLoader import DataLoader, compute_euclidean_distance, prepare_input_img
from network import create_model
import os
import numpy as np
from custom_KFold import MyKFold
from block_matching_utils import find_template_pixel
from PIL import Image
import pandas as pd
from tensorflow import keras
from utils import get_logger, get_default_params
import tensorflow as tf
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate
from joblib import dump, load

'''
Mélanie Bernhardt - ETH Zurich
CLUST Challenge - May 2019

This file defines the main functions of the project: getting the next prediction 
combining the models, train the models, saving the predictions for submission
on the test set.
'''


def get_next_center(k, stop_temporal, c1_prev, c2_prev, img_current,
                    params_dict, model, template_init, c1_init, c2_init, logger=None, est_c1=None, est_c2=None, c1_hist=None, c2_hist=None):
    xax, yax = find_template_pixel(c1_prev, c2_prev,
                                   params_dict['width'], img_current.shape[1], img_current.shape[0])
    template_current = img_current[np.ravel(
        yax), np.ravel(xax)].reshape(1, len(yax), len(xax))
    current_centers = np.asarray([c1_prev, c2_prev]).reshape(1, 2)
    pred = model.predict(
        x=[template_current, template_init, current_centers])
    c1, c2 = pred[0, 0], pred[0, 1]
    if est_c1 is not None and not stop_temporal:
        c1_temp = est_c1.predict(c1_hist.reshape(1, -1))
        c2_temp = est_c2.predict(c2_hist.reshape(1, -1))
        if np.sqrt((c1_temp-c1)**2+(c2_temp-c2)**2) > 1000:
            if logger is None:
                print('WARN: using temporal pred')
            else:
                logger.info('WARN: using temporal pred')
                #logger.info('temp {}, {}'.format(c1_temp, c2_temp))
                #logger.info('net {}, {}'.format(c1, c2))
            c1, c2 = np.mean([c1_temp, c1]), np.mean([c2_temp, c2])
    if ((np.abs(c1-c1_init) > 30) or (np.abs(c2-c2_init) > 30)):
        k += 1
    else:
        k = 0
    if ((k>50)
    or (c1 > img_current.shape[1]) or (c2> img_current.shape[0])
    or (c1 < 0) or (c2<0)):
        #logger.info('WARN: absurd prediction - remove temporal model')
        stop_temporal = True
        #logger.info('keep init')
        c1, c2 = c1_init, c2_init
        k = 0
    try:
        assert (np.abs(c1-c1_prev)<6)
        assert (np.abs(c2-c2_prev)<6)
    except AssertionError:
        print(np.abs(c1-c1_prev))
        print(np.abs(c2-c2_prev))
    return c1, c2, stop_temporal, k


def run_global_cv(fold_iterator, data_dir, checkpoint_dir, logger, params_dict, upsample=True):
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
        model, est_c1, est_c2 = train(traindirs, data_dir, upsample,
                                      params_dict, checkpoint_dir,
                                      logger, validation_generator)
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
                stop_temporal = False
                k = 0 
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
                        c1, c2, stop_temporal, k = get_next_center(k, stop_temporal,
                            c1, c2, img_prev, img_current, params_dict, model, template_init, c1_init, c2_init, logger, est_c1, est_c2, tmp[:, 0], tmp[:, 1])
                    else:
                        c1, c2, stop_temporal, k  = get_next_center(k, stop_temporal,
                            c1, c2, img_prev, img_current, params_dict, model, template_init, c1_init, c2_init, logger)
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
                            # logger.info(
                            #     'Bad dist - maxNCC was {}'.format(maxNCC))
                            logger.info('True {},{}'.format(true[0], true[1]))
                            logger.info('Pred {},{}'.format(
                                c1_orig_coords, c2_orig_coords))
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


def train(traindirs, data_dir, upsample, params_dict, checkpointdir, logger, validation_gen=None):
    if logger is not None:
        logger.info('Training folders are {}'.format(traindirs))
    else:
        print('Training folders are {}'.format(traindirs))
    training_generator = DataLoader(
        data_dir, traindirs, 32,
        width_template=params_dict['width'], upsample=upsample)
    earl = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # Design model
    model = create_model(params_dict['width']+1,
                         params_dict['h1'],
                         params_dict['h2'],
                         params_dict['h3'],
                         embed_size=params_dict['embed_size'],
                         drop_out_rate=params_dict['dropout_rate'],
                         use_batch_norm=params_dict['use_batchnorm'])
    # Train local Net
    if validation_gen is None:
        model.fit_generator(generator=training_generator,
                            use_multiprocessing=True,
                            epochs=params_dict['n_epochs'],
                            workers=4, max_queue_size=20,
                            callbacks=[earl])
    else:
        model.fit_generator(generator=training_generator,
                            validation_data=validation_gen,
                            use_multiprocessing=True,
                            epochs=params_dict['n_epochs'],
                            workers=4, max_queue_size=20)
    if logger is not None:
        logger.info('Local Net trained')
        logger.info('Stopped epoch {}'.format(earl.stopped_epoch))
    else:
        print('Local Net trained')
        print('Stopped epoch {}'.format(earl.stopped_epoch))        
    # Train the temporal model
    for folder in traindirs:
        if logger is not None:
            logger.info('Getting temporal training set for {}'.format(folder))
        else:
            print('Getting temporal training set for {}'.format(folder))
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
        list_imgs = [os.path.join(img_dir, dI)
                     for dI in os.listdir(img_dir)
                     if (dI.endswith('png')
                         and not dI.startswith('.'))]
        n_obs = len(list_imgs)
        X0, X1, X2, X3, X4, X5 = [], [], [], [], [], []
        Y0, Y1, Y2, Y3, Y4, Y5 = [], [], [], [], [], []
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
            Y0 = np.append(Y0, c2_interpolate)
            Y1 = np.append(Y1, c2_interpolate[1:n])
            Y2 = np.append(Y2, c2_interpolate[2:n])
            Y3 = np.append(Y3, c2_interpolate[3:n])
            Y4 = np.append(Y4, c2_interpolate[4:n])
            Y5 = np.append(Y5, c2_interpolate[5:n])
    l = len(X5)
    fullX = np.transpose(np.vstack(
        [X0[0:l], X1[0:l], X2[0:l], X3[0:l], X4[0:l]]))
    fullY = np.transpose(np.vstack(
        [Y0[0:l], Y1[0:l], Y2[0:l], Y3[0:l], Y4[0:l]]))
    c1_label = X5
    c2_label = Y5
    est_c1 = RidgeCV()
    est_c2 = RidgeCV()
    scores_c1 = cross_validate(est_c1, fullX, c1_label, cv=5, scoring=(
        'r2', 'neg_mean_squared_error'))
    scores_c2 = cross_validate(est_c2, fullY, c2_label, cv=5, scoring=(
        'r2', 'neg_mean_squared_error'))
    if logger is not None:
        logger.info('c1')
        logger.info(scores_c1['test_neg_mean_squared_error'])
        logger.info('c2')
        logger.info(scores_c2['test_neg_mean_squared_error'])
    else:
        print('c1')
        print(scores_c1['test_neg_mean_squared_error'])
        print('c2')
        print(scores_c2['test_neg_mean_squared_error'])        
    # Fit on the whole training set
    est_c1.fit(fullX, c1_label)
    est_c2.fit(fullY, c2_label)

    # Save the local Net and the temporal model
    if logger is not None:
        logger.info('Saving trained models to {}'.format(checkpoint_dir))
    else:
        print('Saving trained models to {}'.format(checkpoint_dir))
    model.save_weights(os.path.join(checkpoint_dir, 'model.h5'))
    dump(est_c1, os.path.join(checkpoint_dir, 'est_c1.joblib'))
    dump(est_c2, os.path.join(checkpoint_dir, 'est_c2.joblib'))
    return model, est_c1, est_c2


def predict(testdirs, checkpoint_dir, data_dir, params_dict, upsample=False, resolution_df=None):
    model = create_model(params_dict['width']+1,
                         params_dict['h1'],
                         params_dict['h2'],
                         params_dict['h3'],
                         embed_size=params_dict['embed_size'],
                         drop_out_rate=params_dict['dropout_rate'],
                         use_batch_norm=params_dict['use_batchnorm'])
    model.load_weights(os.path.join(checkpoint_dir, 'model.h5'))
    est_c1 = load(os.path.join(checkpoint_dir, 'est_c1.joblib'))
    est_c2 = load(os.path.join(checkpoint_dir, 'est_c2.joblib'))
    for testfolder in testdirs:
        res_x, res_y = None, None
        if upsample:
            res_x, res_y = resolution_df.loc[resolution_df['scan']
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
            k=0
            stop_temporal=False
            for i in range(2, len(list_imgs)+1):
                if i % 100 == 0:
                    print(i)
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
                    c1, c2, stop_temporal, k = get_next_center(k, stop_temporal,
                        c1, c2, img_current, params_dict, model, template_init, c1_init, c2_init, None, est_c1, est_c2, tmp[:, 0], tmp[:, 1])
                else:
                    c1, c2, stop_temporal, k  = get_next_center(k, stop_temporal,
                        c1, c2, img_current, params_dict, model, template_init, c1_init, c2_init, None)
                # project back in init coords
                if upsample:
                    c1_orig_coords = c1*0.4/res_x
                    c2_orig_coords = c2*0.4/res_y
                else:
                    c1_orig_coords = c1
                    c2_orig_coords = c2
                list_centers = np.append(
                    list_centers, [c1_orig_coords, c2_orig_coords])
            list_centers = list_centers.reshape(-1, 2)
            pred_df = pd.DataFrame()
            pred_df['idx'] = range(1, len(list_centers)+1)
            pred_df['c1'] = list_centers[:, 0]
            pred_df['c2'] = list_centers[:, 1]
            pred_df.to_csv(os.path.join(checkpoint_dir, '{}.txt'.format(
                label_file)), header=False, index=False)

if __name__ == '__main__':
    np.random.seed(seed=42)
    exp_name = 'FINAL_temporal2_epochs15_60-NOJUMP5'
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
    listdir = np.asarray([dI for dI in os.listdir(data_dir) if (
            os.path.isdir(os.path.join(data_dir, dI))
            and not dI == 'feats_matrices')])
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(sess)
    kf = MyKFold(data_dir, n_splits=5)
    fold_iterator = kf.getFolderIterator()
    run_global_cv(fold_iterator, data_dir, checkpoint_dir, logger, params_dict, upsample=False)
    train(listdir, data_dir, False, params_dict, checkpoint_dir, logger, None)