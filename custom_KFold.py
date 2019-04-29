import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from train_local_tracking import get_all_local_features


class MyKFold:
    def __init__(self, data_dir, n_splits=3, width_template=60, bins=20):
        self.resolution_df = pd.read_csv(os.path.join(data_dir, 'resolution.csv'),
                                         sep=',\s+',
                                         decimal='.')
        self.width_template = width_template
        self.n_bins = bins
        self.n_splits = n_splits
        self.data_dir = data_dir
        self.listdir = np.asarray([dI for dI in os.listdir(self.data_dir) if (
            os.path.isdir(os.path.join(data_dir, dI))
            and not dI == 'feats_matrices')])
        self.listdir.sort()
        print(self.listdir.shape)
        num_dir = len(self.listdir)
        if self.n_splits > num_dir:
            print('Warning to many splits reducing to {} splits'.format(num_dir))
            self.n_splits = num_dir
            self.dir_per_fold = 1
        else:
            self.dir_per_fold = num_dir//self.n_splits
        print('Number folders per fold: {}'.format(self.dir_per_fold))

    def getDataIterator(self):
        """for the estimators that need features"""
        p = 0
        print(len(self.listdir))
        for fold in range(self.n_splits):
            if fold == (self.n_splits-1):
                test_indices = np.arange(
                    p, len(self.listdir), dtype='int')
            else:
                test_indices = np.arange(
                    p, p+self.dir_per_fold, dtype='int')
            print(p)
            test_dirs = self.listdir[test_indices]
            print(test_dirs)
            train_dirs = np.delete(self.listdir, test_indices)
            print(train_dirs)
            p += self.dir_per_fold
            X_train, x_train, y_train, _, _ = get_all_local_features(
                train_dirs, self.data_dir, self.width_template, self.n_bins)
            X_test, x_test, y_test, res_test_x, res_test_y = get_all_local_features(
                test_dirs, self.data_dir, self.width_template, self.n_bins, self.resolution_df)
            yield X_train, X_test, x_train, x_test, y_train, y_test, res_test_x, res_test_y

    def getFolderIterator(self):
        p = 0
        print(len(self.listdir))
        listdir = np.random.permutation(self.listdir)
        for fold in range(self.n_splits):
            if fold == (self.n_splits-1):
                test_indices = np.arange(
                    p, len(self.listdir), dtype='int')
            else:
                test_indices = np.arange(
                    p, p+self.dir_per_fold, dtype='int')
            test_dirs = listdir[test_indices]
            print(test_dirs)
            train_dirs = np.delete(listdir, test_indices)
            print(train_dirs)
            p += self.dir_per_fold
            yield train_dirs, test_dirs


if __name__ == '__main__':
    np.random.seed(seed=42)
    data_dir = os.getenv('DATA_PATH')
    print(data_dir)
    kf = MyKFold(data_dir, n_splits=5)
    iterator = kf.getDataIterator()
    mse_x = []
    mse_y = []
    dist = []
    done = False
    while not done:
        try:
            X_train, X_test, x_train, x_test, y_train, y_test, res_test_x, res_test_y = next(
                iterator)
            print(len(X_train))
            est_x = RandomForestRegressor(n_estimators=1000)
            est_x.fit(X_train, x_train.ravel())
            est_y = RandomForestRegressor(n_estimators=1000)
            est_y.fit(X_train, y_train.ravel())
            x_pred = est_x.predict(X_test)
            y_pred = est_y.predict(X_test)
            d = np.mean(
                np.sqrt(((x_pred - x_test)*res_test_x)**2 +
                        ((y_pred - y_test)*res_test_y)**2)
            )
            print(d)
            print(np.mean((x_pred - x_test)**2))
            print(np.mean((y_pred - y_test)**2))
            mse_x.append(np.mean((x_pred - x_test)**2))
            mse_y.append(np.mean((y_pred - y_test)**2))
            dist.append(d)
        except StopIteration:
            done = True
    print('Mean MSE for x across fold {}, std {}'.format(
        np.mean(mse_x), np.std(mse_x)))
    print('Mean MSE for y across fold {}, std {}'.format(
        np.mean(mse_y), np.std(mse_y)))
    print('Mean Euclidian distance across fold {}, std {}'.format(
        np.mean(dist), np.std(dist)))
