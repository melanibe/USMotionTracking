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
                                   params_dict['width'], img_current.shape[1], img_current.shape[0])
    template_current = img_current[np.ravel(
        yax), np.ravel(xax)].reshape(1, len(yax), len(xax))
    current_centers = np.asarray([c1, c2]).reshape(1, 2)
    pred = model.predict(
        x=[template_current, template_init, current_centers])
    old_c1, old_c2 = c1, c2
    c1, c2 = pred[0, 0], pred[0, 1]
    if np.sqrt((c1_prev-c1)**2+(c2_prev-c2)**2) > 15:
        if np.sqrt((old_c1-c1_prev)**2+(old_c2-c2_prev)**2) < np.sqrt((old_c1-c1)**2+(old_c2-c2)**2):
            if logger is None:
                print('WARN: VERY weird prediction mean maxNCC old_pred')
            else:
                logger.info(
                    'WARN: VERY weird prediction mean both maxNCC old_pred')
                logger.info('previous {},{}'.format(c1_prev, c2_prev))
                logger.info('NCC {},{}'.format(old_c1, old_c2))
                logger.info('proposed by net {},{}'.format(c1, c2))
                logger.info('kept {},{}'.format(
                    (old_c1+c1_prev)/2, (old_c2+c2_prev)/2))
            c1, c2 = (old_c1+c1_prev)/2, (old_c2+c2_prev)/2
        else:
            if logger is None:
                print('WARN: weird prediction mean both maxNCC current_pred')
            else:
                logger.info(
                    'WARN: weird prediction mean both maxNCC current_pred')
                logger.info('previous {},{}'.format(c1_prev, c2_prev))
                logger.info('NCC {},{}'.format(old_c1, old_c2))
                logger.info('proposed by net {},{}'.format(c1, c2))
                logger.info('kept {},{}'.format((old_c1+c1)/2, (old_c2+c2)/2))
            c1, c2 = (old_c1+c1)/2, (old_c2+c2)/2
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
            img_init = prepare_input_img(img_init, res_x, res_y, upsample)
            X0 = []
            X1 = []
            X2 = []
            X3 = []
            X4 = []
            X5 = []
            X6 = []
            X7 = []
            X8 = []
            X9 = []
            X10 = []
            for label in list_label_files:
                df = predict_feature(label, img_init,
                                    img_dir, res_x, res_y, model, annotation_dir, params_dict, checkpoint_dir, upsample)
                n = len(df.c1.values)
                X0 = np.append(X0, df.c1.values)
                X1 = np.append(X1, df.c1.values[1:n])
                X2 = np.append(X2, df.c1.values[2:n])
                X3 = np.append(X3, df.c1.values[3:n])
                X4 = np.append(X4, df.c1.values[4:n])
                X5 = np.append(X5, df.c1.values[5:n])
                X6 = np.append(X6, df.c1.values[6:n])
                X7 = np.append(X7, df.c1.values[7:n])
                X8 = np.append(X8, df.c1.values[8:n])
                X9 = np.append(X9, df.c1.values[9:n])
                X10 = np.append(X10, df.c1.values[10:n])
        l = len(X10)
        fullX = np.transpose(np.vstack([X0[0:l], X1[0:l], X2[0:l], X3[0:l], X4[0:l], X5[0:l], X6[0:l], X7[0:l], X8[0:l], X9[0:l]]))
        y = X10
        est = RidgeCV()
        scores = cross_validate(est, fullX, y, cv=5, scoring=('r2', 'neg_mean_squared_error'))
        logger.info(scores['test_neg_mean_squared_error'])
        est.fit(fullX, y)
            
        # PREDICT WITH GLOBAL MATCHING + LOCAL MODEL ON TEST SET
        curr_fold_dist = []
        curr_fold_pix = []
        for k, testfolder in enumerate(testdirs):
            if k == 0:  # TODO just for debug
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
                    img_dir, res_x, res_y, model, annotation_dir, params_dict, checkpoint_dir, upsample):
    list_imgs = [os.path.join(img_dir, dI)
                for dI in os.listdir(img_dir)
                if (dI.endswith('png')
                    and not dI.startswith('.'))]
    n_obs = len(list_imgs)
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
        c1, c2, old_c1, old_c2, maxNCC = get_next_center(
            c1, c2, img_prev, img_current, params_dict, model, template_init)
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
    pred_df.to_csv(os.path.join(checkpoint_dir, '{}'.format(
        label_file)), header=False, index=False)
    print('{} DONE'.format(label_file))
    return pred_df


if __name__ == '__main__':
    np.random.seed(seed=42)
    exp_name = '2layers_noup_se20'
    params_dict = {'dropout_rate': 0.5, 'n_epochs': 10,
                   'h3': 0, 'embed_size': 256, 'width': 60, 'search_w': 20}

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
