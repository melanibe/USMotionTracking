import os
from PIL import Image
import numpy as np
import pandas as pd
from block_matching_utils import find_template_pixel
from tensorflow import keras
import tensorflow as tf
def metrics_distance(labels, preds):
    return compute_euclidean_distance_tf(None, "", preds, labels)

def compute_euclidean_distance_tf(res_df, exp_name, preds, labels):
    if res_df is not None:
        curr_res_x = res_df.loc[res_df['scan'] == exp_name, 'res_x'].values[0]
        curr_res_y = res_df.loc[res_df['scan'] == exp_name, 'res_y'].values[0]
    else:
        curr_res_x = 1
        curr_res_y = 1
    return tf.reduce_mean(
        tf.sqrt(((preds[:, 0] - labels[:, 0])*curr_res_x)**2 +
                ((preds[:, 1] - labels[:, 1])*curr_res_y)**2)
    )

def compute_euclidean_distance(res_df, exp_name, preds, labels):
    if res_df is not None:
        curr_res_x = res_df.loc[res_df['scan'] == exp_name, 'res_x'].values[0]
        curr_res_y = res_df.loc[res_df['scan'] == exp_name, 'res_y'].values[0]
    else:
        curr_res_x = 1
        curr_res_y = 1
    return np.mean(
        np.sqrt(((preds[:, 0] - labels[:, 0])*curr_res_x)**2 +
                ((preds[:, 1] - labels[:, 1])*curr_res_y)**2)
    )


class DataLoader(keras.utils.Sequence):
    def __init__(self,
                 data_dir,
                 list_dir,
                 batch_size,
                 width_template=60,
                 resolution_df=None,
                 shuffle=True,
                 type='train'):
        self.type = type
        self.data_dir = data_dir
        self.list_dir = list_dir
        self.batch_size = batch_size
        self.width_template = width_template
        self.resolution_df = None
        self.list_imgs = []
        self.list_init_x = []
        self.list_init_y = []
        self.orig_labels_x = []
        self.orig_labels_y = []
        self.list_imgs_init = []
        self.resolution_df = pd.read_csv(os.path.join(data_dir, 'resolution.csv'),
                                         sep=',\s+',
                                         decimal='.')
        for subfolder in self.list_dir:
            current_dir = os.path.join(self.data_dir, subfolder)
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
            for label_file in list_label_files:
                df = pd.read_csv(label_file,
                                 header=None,
                                 names=['id', 'x', 'y'],
                                 sep='\s+')
                n_obs = len(df)
                try:
                    Image.open(os.path.join(self.data_dir,
                                            subfolder, 'Data', "0001.png"))
                    self.list_imgs = np.append(self.list_imgs,
                                               [os.path.join(self.data_dir, subfolder, 'Data', "{:04d}.png".format(int(i)))
                                                for i in df.id.values[1:n_obs]]
                                               )
                    self.list_imgs_init = np.append(self.list_imgs_init,
                                                    np.repeat(os.path.join(self.data_dir,
                                                                           subfolder, 'Data', "0001.png"), n_obs-1)
                                                    )
                except FileNotFoundError:
                    self.list_imgs = np.append(self.list_imgs,
                                               [os.path.join(self.data_dir, subfolder, 'Data', "{:05d}.png".format(int(i)))
                                                for i in df.id.values[1:n_obs]]
                                               )
                    self.list_imgs_init = np.append(self.list_imgs_init,
                                                    np.repeat(os.path.join(self.data_dir,
                                                                           subfolder, 'Data',  "00001.png"), n_obs-1)
                                                    )

                self.list_init_x = np.append(
                    self.list_init_x, np.repeat(df.x.values[0], n_obs-1))
                self.list_init_y = np.append(
                    self.list_init_y, np.repeat(df.y.values[0], n_obs-1))
                self.orig_labels_x = np.append(
                    self.orig_labels_x, df.x.values[1:n_obs])
                self.orig_labels_y = np.append(
                    self.orig_labels_y, df.y.values[1:n_obs])
        self.shuffle = shuffle
        self.u_x_list = np.random.randn(len(self.orig_labels_x))*4
        self.u_y_list = np.random.randn(len(self.orig_labels_y))*4
        if self.type == 'val':
            self.shuffle = False  # don't shuffle if test set.
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.orig_labels_x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        [batch_imgs, batch_imgs_init,
            batch_centers], batch_labels = self.__data_generation(indexes)
        return [batch_imgs, batch_imgs_init, batch_centers], batch_labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.orig_labels_x))
        if type == 'train':
            self.u_x_list = np.random.randn(len(self.orig_labels_y))*5
            self.u_y_list = np.random.randn(len(self.orig_labels_y))*5
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        batch_orig_centers = np.zeros((len(indexes), 2))
        batch_centers = np.zeros((len(indexes), 2))
        batch_orig_centers[:, 0] = self.orig_labels_x[indexes]
        batch_orig_centers[:, 1] = self.orig_labels_y[indexes]
        #batch_init_x = self.list_init_x[indexes]
        #batch_init_y = self.list_init_y[indexes]
        batch_labels = np.zeros((len(indexes), 2))
        batch_imgs = np.zeros(
            (len(indexes),
                self.width_template+1,
                self.width_template+1)
        )
        batch_imgs_init = np.zeros(
            (len(indexes),
                self.width_template+1,
                self.width_template+1)
        )

        for i, idx in enumerate(indexes):
            img = np.asarray(Image.open(self.list_imgs[idx]))
            mean = np.mean(img)
            sd = np.std(img)
            img = (img-mean)/sd
            img_init = np.asarray(Image.open(self.list_imgs_init[idx]))
            mean = np.mean(img_init)
            sd = np.std(img_init)
            img_init = (img_init-mean)/sd          
            c1_init = self.list_init_x[idx]
            c2_init = self.list_init_y[idx]
            xax, yax = find_template_pixel(
                c1_init,
                c2_init,
                width=self.width_template)
            batch_imgs_init[i, :, :] = img_init[np.ravel(yax), np.ravel(xax)].reshape(self.width_template+1,
                                                                                      self.width_template+1)
            # true location
            c1, c2 = self.orig_labels_x[idx], self.orig_labels_y[idx]
            # perturbed center of the template
            # N(0,10) i.e. 95% perturbation are in -20;20.
            u_x = self.u_x_list[idx]
            u_y = self.u_y_list[idx]
            c1_perturbed = c1 - u_x
            c2_perturbed = c2 - u_y
            batch_centers[i, 0] = c1_perturbed
            batch_centers[i, 1] = c2_perturbed
            # labels is the coord wrt to the center of
            # the pixel so here c1 = c1_perturbed - 2
            # label_x = -2 i.e. c1 = c1_perturbed + label
            batch_labels[i, 0] = c1
            batch_labels[i, 1] = c2
            xax, yax = find_template_pixel(
                c1_perturbed,
                c2_perturbed,
                width=self.width_template)
            batch_imgs[i] = img[np.ravel(yax), np.ravel(xax)].reshape(self.width_template+1,
                                                                      self.width_template+1)
        return([batch_imgs, batch_imgs_init, batch_centers], batch_labels)
