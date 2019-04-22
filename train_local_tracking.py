import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from block_matching_utils import find_template_pixel
from sklearn.model_selection import KFold


def get_local_features(data_dir, width_template=60, bins=20):
    listdir = [dI for dI in os.listdir(
        data_dir) if (
            os.path.isdir(os.path.join(data_dir, dI))
            and not dI == 'feats_matrices')]
    print(listdir)
    listdir.sort()
    feats_intensity = None
    labels_x = None
    labels_y = None
    # x_init = None
    # y_init = None
    for subfolder in listdir:
        current_dir = os.path.join(data_dir, subfolder)
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
            print('id length {}'.format(len(df.id.values)))
            for i in df.id.values:
                img = np.asarray(Image.open(list_imgs[int(i)-1]))
                c1, c2 = df.loc[df['id'] == i, ['x', 'y']].values[0, :]
                xax, yax = find_template_pixel(c1, c2,
                                               width=width_template)
                img_template = img[np.ravel(yax), np.ravel(xax)]
                tmp, _ = np.histogram(img_template, bins=bins, range=(0, 255))
                if feats_intensity is not None:
                    feats_intensity = np.concatenate(
                        (feats_intensity, tmp.reshape(1, bins)), axis=0)
                else:
                    feats_intensity = np.reshape(tmp, (1, bins))
            if labels_x is not None:
                labels_x = np.concatenate(
                    (labels_x, df.x.values.reshape(-1, 1)), axis=0)
                labels_y = np.concatenate(
                    (labels_y, df.y.values.reshape(-1, 1)), axis=0)
               # x_init = np.concatenate(
               #     (x_init, np.repeat(df.x.values[0], len(df.id.values))))
               # y_init = np.concatenate(
               #     (y_init, np.repeat(df.y.values[0], len(df.id.values))))
                feats_init = np.concatenate(
                    (feats_init,
                        np.tile(current_feats_init, len(df.id.values)).reshape(-1, bins)),
                    axis=0)
                print('feats init shape {}'.format(feats_init.shape))
                # print('x shape {}'.format(x_init.shape))
                print('x labels {}'.format(labels_x.shape))
                print('feats shape {}'.format(feats_intensity.shape))
            else:
                labels_x = df.x.values.reshape(-1, 1)
                labels_y = df.y.values.reshape(-1, 1)
                #x_init = np.repeat(df.x.values[0], len(df.id.values))
                #y_init = np.repeat(df.y.values[0], len(df.id.values))
                feats_init = np.tile(current_feats_init,
                                     len(df.id.values)).reshape(-1, bins)
    X_full = np.concatenate((feats_intensity,
                             feats_init), axis=1)
    #x_init.reshape((-1, 1)),
    #y_init.reshape((-1, 1)),
    return X_full, labels_x, labels_y


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
    X_full_1, labels_x, labels_y = get_local_features(data_dir,
                                                      width_template=60,
                                                      bins=20)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_60_20'), X_full_1)
    np.save(os.path.join(data_dir, 'feats_matrices', 'labels_x'), labels_x)
    np.save(os.path.join(data_dir, 'feats_matrices', 'labels_y'), labels_y)
    X_full_2, labels_x_2, labels_y_2 = get_local_features(data_dir,
                                                          width_template=60,
                                                          bins=40)
    print(np.max(np.abs(labels_x_2-labels_x)))
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_60_40'), X_full_2)
    X_full_3, _, _ = get_local_features(data_dir,
                                        width_template=60,
                                        bins=10)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_60_10'), X_full_3)
    X_full_4, _, _ = get_local_features(data_dir,
                                        width_template=50,
                                        bins=20)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_50_20'), X_full_4)
    X_full_5, _, _ = get_local_features(data_dir,
                                        width_template=50,
                                        bins=40)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_50_40'), X_full_5)
    X_full_6, _, _ = get_local_features(data_dir,
                                        width_template=80,
                                        bins=20)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_80_20'), X_full_6)
    X_full_7, _, _ = get_local_features(data_dir,
                                        width_template=80,
                                        bins=40)
    np.save(os.path.join(data_dir, 'feats_matrices', 'X_full_80_40'), X_full_7)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meansquare_error = np.zeros((8,5))
    for i in range(7):
        est = RandomForestRegressor(n_estimators=1000)
        X = eval('X_full_{}'.format(i+1))
        y = labels_x
        c = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            est.fit(X=X_train, y=y_train)
            y_pred = est.predict(X_test)
            meansquare_error[i, c] = np.mean((y_pred - y_test)**2)
            print(np.max(np.abs(y_pred - y_test)))
    print(np.mean(meansquare_error, axis=1))
    print(np.std(meansquare_error, axis=1))
    print(np.max(meansquare_error, axis=1))
    print(np.min(meansquare_error, axis=1))