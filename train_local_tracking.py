import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from block_matching_utils import find_template_pixel
from sklearn.model_selection import KFold


def get_local_features(current_dir, width_template=60, bins=20):
    feats_intensity = None
    labels_x = []
    labels_y = []
    feats_init = None
    annotation_dir = os.path.join(current_dir, 'Annotation')
    img_dir = os.path.join(current_dir, 'Data')
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
        df = pd.read_csv(label_file,
                         header=None,
                         names=['id', 'x', 'y'],
                         sep='\s+')
        c1_init, c2_init = df.loc[df['id'] == 1, ['x', 'y']].values[0, :]
        xax, yax = find_template_pixel(c1_init, c2_init,
                                       width=width_template)
        init_template = img_init[np.ravel(yax), np.ravel(xax)]
        current_feats_init, _ = np.histogram(
            init_template, bins=bins, range=(0, 255))
        n_obs = len(df.id.values)
        print('id length {}'.format(n_obs))
        for i in df.id.values[1:n_obs]:
            img = np.asarray(Image.open(list_imgs[int(i)-1]))
            # true location
            c1, c2 = df.loc[df['id'] == i, ['x', 'y']].values[0, :]
            # perturbed center of the template
            u_x = np.random.rand()*40-20
            u_y = np.random.rand()*40-20
            c1_perturbed = c1 - u_x # ~ Unif(-20,20)
            c2_perturbed = c2 - u_y
            # labels is the coord wrt to the center of
            # the pixel so here c1 = c1_perturbed - 2
            # label_x = -2 i.e. c1 = c1_perturbed + label
            labels_x = np.append(labels_x, u_x)
            labels_y = np.append(labels_y, u_y)
            xax, yax = find_template_pixel(c1_perturbed, c2_perturbed,
                                           width=width_template)
            img_template = img[np.ravel(yax), np.ravel(xax)]
            tmp, _ = np.histogram(img_template, bins=bins, range=(0, 255))
            if feats_intensity is not None:
                feats_intensity = np.concatenate(
                    (feats_intensity, tmp.reshape(1, bins)), axis=0)
            else:
                feats_intensity = np.reshape(tmp, (1, bins))
        if feats_init is not None:
            feats_init = np.concatenate(
                (feats_init,
                    np.tile(current_feats_init, n_obs-1).reshape(-1, bins)),
                axis=0)
            print('feats init shape {}'.format(feats_init.shape))
            print('x labels {}'.format(labels_x.shape))
            print('feats shape {}'.format(feats_intensity.shape))
        else:
            feats_init = np.tile(current_feats_init,
                                 n_obs-1).reshape(-1, bins)
    X_full = np.concatenate((feats_intensity,
                             feats_init), axis=1)
    return X_full, labels_x, labels_y


def get_all_local_features(listdir, data_dir, width_template=60, bins=20, res_df = None):
    X_full = None
    labels_x = None
    labels_y = None
    res_x = None
    res_y = None
    for subfolder in listdir:
        try:
            current_X = np.load(os.path.join(data_dir, subfolder, 'X_{}_{}.npy'.format(width_template, bins)))
            current_x = np.load(os.path.join(data_dir, subfolder, 'x_coords.npy'))
            current_y = np.load(os.path.join(data_dir, subfolder, 'y_coords.npy'))
        except FileNotFoundError:
            current_X, current_x, current_y = get_local_features(
                subfolder, width_template, bins)
            np.save(os.path.join(data_dir, subfolder, 'X_{}_{}.npy'.format(width_template, bins)), current_X)
            np.save(os.path.join(data_dir, subfolder, 'x_coords.npy'), current_x)
            np.save(os.path.join(data_dir, subfolder, 'y_coords.npy'), current_y)
        assert(len(current_X)==len(current_x)==len(current_y))
        if res_df is not None:
            curr_res_x = res_df.loc[res_df['scan']==subfolder, 'res_x'].values[0]
            curr_res_y = res_df.loc[res_df['scan']==subfolder, 'res_y'].values[0]
            curr_res_x = np.repeat(curr_res_x, len(current_X))
            curr_res_y = np.repeat(curr_res_y, len(current_X))
        if X_full is not None:
            X_full = np.concatenate((X_full, current_X), axis=0)
            labels_x = np.append(labels_x, current_x)
            labels_y = np.append(labels_y, current_y)
            if res_df is not None:
                res_x = np.append(res_x, curr_res_x)
                res_y = np.append(res_y, curr_res_y)
        else:
            X_full = current_X
            labels_x = current_x
            labels_y = current_y
            if res_df is not None:
                res_x = curr_res_x
                res_y = curr_res_y
    try:
        print(res_x.shape)
    except AttributeError:
        pass
    return X_full, labels_x, labels_y, res_x, res_y


if __name__ == '__main__':
    data_dir = os.getenv('DATA_PATH')
    print(data_dir)
    """
    try:
        X_full = np.load(os.path.join(data_dir, 'feats_matrices', 'X_full.npy'))
        labels_x = np.load(os.path.join(data_dir, 'feats_matrices', 'labels_x.npy'))
        labels_y = np.load(os.path.join(data_dir, 'feats_matrices', 'labels_y.npy'))
    except FileNotFoundError:
        X_full, labels_x, labels_y = get_local_features(data_dir, width_template=60)
        np.save(os.path.join(data_dir, 'feats_matrices', 'X_full'), X_full)
        np.save(os.path.join(data_dir, 'feats_matrices', 'labels_x'), labels_x)
        np.save(os.path.join(data_dir, 'feats_matrices', 'labels_y'), labels_y)
    X_train, X_test, y_train, y_test = train_test_split(X_full,
                                                        labels_x.ravel(),
                                                        test_size=0.33,
                                                        random_state=42)
    """
    listdir = [os.path.join(data_dir, dI) for dI in os.listdir(
        data_dir) if (
        os.path.isdir(os.path.join(data_dir, dI))
        and not dI == 'feats_matrices')]
    X_full_1, labels_x, labels_y = get_local_features(listdir,
                                                      width_template=60,
                                                      bins=20)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_60_20'), X_full_1)
    np.save(os.path.join(data_dir, 'feats_matrices', 'labels_x'), labels_x)
    np.save(os.path.join(data_dir, 'feats_matrices', 'labels_y'), labels_y)
    X_full_2, labels_x_2, labels_y_2 = get_local_features(listdir,
                                                          width_template=60,
                                                          bins=40)
    print(np.max(np.abs(labels_x_2-labels_x)))
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_60_40'), X_full_2)
    X_full_3, _, _ = get_local_features(listdir,
                                        width_template=60,
                                        bins=10)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_60_10'), X_full_3)
    X_full_4, _, _ = get_local_features(listdir,
                                        width_template=50,
                                        bins=20)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_50_20'), X_full_4)
    X_full_5, _, _ = get_local_features(listdir,
                                        width_template=50,
                                        bins=40)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_50_40'), X_full_5)
    X_full_6, _, _ = get_local_features(listdir,
                                        width_template=80,
                                        bins=20)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_80_20'), X_full_6)
    X_full_7, _, _ = get_local_features(listdir,
                                        width_template=80,
                                        bins=40)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_80_40'), X_full_7)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meansquare_error = np.zeros((8, 5))
    for i in range(7):
        X = eval('X_full_{}'.format(i+1))
        y = labels_x
        c = 0
        for train_index, test_index in kf.split(X):
            est = RandomForestRegressor(n_estimators=1000)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            est.fit(X=X_train, y=y_train.ravel())
            y_pred = est.predict(X_test)
            meansquare_error[i, c] = np.mean((y_pred - y_test)**2)
            print(np.max(np.abs(y_pred - y_test)))
    print(np.mean(meansquare_error, axis=1))
    print(np.std(meansquare_error, axis=1))
    print(np.max(meansquare_error, axis=1))
    print(np.min(meansquare_error, axis=1))
