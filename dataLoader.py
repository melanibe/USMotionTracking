import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from block_matching_utils import find_template_pixel
from tensorflow import keras
import tensorflow as tf
import sys
import skimage
import parmap
'''
Mélanie Bernhardt - ETH Zurich
CLUST Challenge
'''
def return_orig_pairs(idx, df, data_dir, subfolder):
    init_c1, init_c2 = df.x_newres.values[0], df.y_newres.values[0]
    try:
        Image.open(os.path.join(data_dir,
                                subfolder, 'Data', "0001.png"))
        img = os.path.join(data_dir, subfolder, 'Data', "{:04d}.png".format(int(idx)))
        imgs_init = os.path.join(data_dir, subfolder, 'Data', "{:04d}.png".format(int(1)))
    except FileNotFoundError:
        img = os.path.join(data_dir, subfolder, 'Data', "{:05d}.png".format(int(idx)))
        imgs_init = os.path.join(data_dir, subfolder, 'Data', "{:05d}.png".format(int(1)))
    c1, c2 = df.loc[df['id']==idx, ['x_newres', 'y_newres']].values[0]
    return c1, c2, init_c1, init_c2, imgs_init, img


def return_rdm_pairs(idx, max_j, df, data_dir, subfolder):
    if max_j==1:
        j = 0
    else:
        j = int(np.random.choice(np.arange(1, max_j)))
    init_c1, init_c2 = df.x_newres.values[j], df.y_newres.values[j]
    try:
        Image.open(os.path.join(data_dir,
                                subfolder, 'Data', "0001.png"))
        img = os.path.join(data_dir, subfolder, 'Data', "{:04d}.png".format(int(idx)))
        imgs_init = os.path.join(data_dir, subfolder, 'Data', "{:04d}.png".format(int(df.id.values[j])))
    except FileNotFoundError:
        img = os.path.join(data_dir, subfolder, 'Data', "{:05d}.png".format(int(idx)))
        imgs_init = os.path.join(data_dir, subfolder, 'Data', "{:05d}.png".format(int(df.id.values[j])))
    c1, c2 = df.loc[df['id']==idx, ['x_newres', 'y_newres']].values[0]
    return c1, c2, init_c1, init_c2, imgs_init, img

def metrics_distance(labels, preds):
    curr_res_x = 0.4
    curr_res_y = 0.4
    return tf.reduce_mean(
        tf.sqrt(((preds[:, 0] - labels[:, 0])*curr_res_x)**2 +
                ((preds[:, 1] - labels[:, 1])*curr_res_y)**2)
    )


def compute_euclidean_distance(preds, labels):
    curr_res_x = 0.4
    curr_res_y = 0.4
    return np.mean(
        np.sqrt(((preds[:, 0] - labels[:, 0])*curr_res_x)**2 +
                ((preds[:, 1] - labels[:, 1])*curr_res_y)**2)
    )


def prepare_input_img(img, res_x=None, res_y=None, upsample=False):
    if upsample:
        img = skimage.transform.resize(
            img, (int(np.floor(img.shape[0]*res_y/0.4)),
                int(np.floor(img.shape[1]*res_x/0.4))))
    img = img/255.0
    return img


class DataLoader(keras.utils.Sequence):
    def __init__(self,
                 data_dir,
                 list_dir,
                 batch_size,
                 width_template=60,
                 resolution_df=None,
                 shuffle=True,
                 type='train',
                 upsample=True):
        self.upsample = upsample
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
        self.list_res_x = []
        self.list_res_y = []
        self.resolution_df = pd.read_csv(os.path.join(data_dir, 'resolution.csv'),
                                         sep=',\s+',
                                         decimal='.')
        for subfolder in self.list_dir:
            res_x = self.resolution_df.loc[self.resolution_df['scan']
                                           == subfolder, 'res_x'].values[0]
            res_y = self.resolution_df.loc[self.resolution_df['scan']
                                           == subfolder, 'res_y'].values[0]
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
                if self.upsample:
                    df['x_newres'] = df['x']*res_x/0.4
                    df['y_newres'] = df['y']*res_y/0.4
                else:
                    df['x_newres'] = df['x']
                    df['y_newres'] = df['y']
                c1_init, c2_init = df[['x_newres', 'y_newres']].values[0]
                try:
                    img_init = np.asarray(Image.open(os.path.join(data_dir, subfolder, 'Data', "{:04d}.png".format(int(1)))))
                except FileNotFoundError:
                    img_init = np.asarray(Image.open(os.path.join(data_dir, subfolder, 'Data', "{:05d}.png".format(int(1)))))
                if np.isnan(c1_init):
                    print(label_file)
                    print(df.head())
                    print(df[['x', 'y']])
                    print(df[['x_newres', 'y_newres']])
                    print(c1_init, c2_init)
                    print(img_init.shape[1], img_init.shape[0])
                xax, yax = find_template_pixel(c1_init, c2_init, 300, img_init.shape[1], img_init.shape[0]) 
                template_big = img_init[np.ravel(yax), np.ravel(xax)]    
                listid = df.id.values[1:n_obs].astype(int).tolist()
                big_array = np.asarray(parmap.map(return_orig_pairs, listid, df, self.data_dir, subfolder))
                if self.type == 'train':
                    other_big_array = np.asarray(parmap.starmap(return_rdm_pairs, zip(listid, range(1, n_obs)), df, self.data_dir, subfolder))
                    other_big_array2 = np.asarray(parmap.starmap(return_rdm_pairs, zip(listid, range(1, n_obs)), df, self.data_dir, subfolder))
                    other_big_array3 = np.asarray(parmap.starmap(return_rdm_pairs, zip(listid, range(1, n_obs)), df, self.data_dir, subfolder))
                    self.orig_labels_x = np.append(self.orig_labels_x, [big_array[:, 0].astype(float), other_big_array[:, 0].astype(float), other_big_array2[:, 0].astype(float), other_big_array3[:, 0].astype(float)])
                    self.orig_labels_y = np.append(self.orig_labels_y, [big_array[:, 1].astype(float), other_big_array[:, 1].astype(float), other_big_array2[:, 1].astype(float), other_big_array3[:, 1].astype(float)])
                    self.list_init_x = np.append(self.list_init_x, [big_array[:, 2].astype(float), other_big_array[:, 2].astype(float), other_big_array2[:, 2].astype(float), other_big_array3[:, 2].astype(float)])
                    self.list_init_y = np.append(self.list_init_y, [big_array[:, 3].astype(float), other_big_array[:, 3].astype(float), other_big_array2[:, 3].astype(float), other_big_array3[:, 3].astype(float)])
                    self.list_imgs = np.append(self.list_imgs, [big_array[:, 5], other_big_array[:, 5], other_big_array2[:, 5], other_big_array3[:, 5]])
                    self.list_imgs_init = np.append(self.list_imgs_init, [big_array[:, 4], other_big_array[:, 4], other_big_array2[:, 4], other_big_array3[:, 4]])
                    self.list_res_x = np.append(
                        self.list_res_x, np.repeat(res_x, 4*(n_obs-1)))
                    self.list_res_y = np.append(
                        self.list_res_y, np.repeat(res_y, 4*(n_obs-1)))
                else:
                    self.orig_labels_x = np.append(self.orig_labels_x, big_array[:, 0].astype(float))
                    self.orig_labels_y = np.append(self.orig_labels_y, big_array[:, 1].astype(float))
                    self.list_init_x = np.append(self.list_init_x, big_array[:, 2].astype(float))
                    self.list_init_y = np.append(self.list_init_y, big_array[:, 3].astype(float))
                    self.list_imgs = np.append(self.list_imgs, big_array[:, 5])
                    self.list_imgs_init = np.append(self.list_imgs_init, big_array[:, 4])
                    self.list_res_x = np.append(
                        self.list_res_x, np.repeat(res_x, (n_obs-1)))
                    self.list_res_y = np.append(
                        self.list_res_y, np.repeat(res_y, (n_obs-1)))
        print(self.list_imgs.shape)
        print(self.orig_labels_x.shape)
        self.shuffle = shuffle
        self.u_x_list = np.random.randn(len(self.orig_labels_x))*5
        self.u_y_list = np.random.randn(len(self.orig_labels_y))*5
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
        print(len(self.orig_labels_x))
        self.indexes = np.arange(len(self.orig_labels_x))
        if type == 'train':
            self.u_x_list = np.random.randn(len(self.orig_labels_y))*5
            self.u_y_list = np.random.randn(len(self.orig_labels_y))*5
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        batch_orig_centers = np.zeros((len(indexes), 2))
        batch_centers = np.zeros((len(indexes), 2))
        batch_orig_centers[:, 0] = self.orig_labels_x[indexes]
        batch_orig_centers[:, 1] = self.orig_labels_y[indexes]
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
        try:
            for i, idx in enumerate(indexes):
                img = np.asarray(Image.open(self.list_imgs[idx]))
                img = prepare_input_img(img, self.list_res_x[idx],
                                        self.list_res_y[idx], self.upsample)
                img_init = np.asarray(Image.open(self.list_imgs_init[idx]))
                img_init = prepare_input_img(
                    img_init, self.list_res_x[idx],
                    self.list_res_y[idx], self.upsample)
                c1_init = self.list_init_x[idx]
                c2_init = self.list_init_y[idx]
                # print(c1_init, c2_init)
                xax, yax = find_template_pixel(
                    c1_init,
                    c2_init,
                    self.width_template, img_init.shape[1], img_init.shape[0])
                try:
                    batch_imgs_init[i, :, :] = img_init[np.ravel(yax), np.ravel(xax)].reshape(self.width_template+1,
                                                                                          self.width_template+1)
                except:
                    print(c1_init, c2_init)
                    print(self.list_imgs_init[idx])
                # true location
                c1, c2 = self.orig_labels_x[idx], self.orig_labels_y[idx]
                # perturbed center of the template
                # N(0,10) i.e. 95% perturbation are in -20;20.
                u_x, u_y = self.u_x_list[idx], self.u_y_list[idx]
                c1_perturbed, c2_perturbed = c1 - u_x, c2 - u_y
                batch_centers[i] = [c1_perturbed, c2_perturbed]
                # labels is the coord wrt to the center of
                # the pixel so here c1 = c1_perturbed - 2
                # label_x = -2 i.e. c1 = c1_perturbed + label
                batch_labels[i] = [c1, c2]
                xax, yax = find_template_pixel(
                    c1_perturbed,
                    c2_perturbed,
                    self.width_template, img_init.shape[1], img_init.shape[0])
                batch_imgs[i] = img[np.ravel(yax), np.ravel(xax)].reshape(self.width_template+1,
                                                                          self.width_template+1)
        except:
            print('ERROR')
            print(self.list_res_x[idx], self.list_res_y[idx])
            print(i)
            print(idx)
            print(c1, c2)
            print(np.max(np.ravel(yax)))
            print(np.max(np.ravel(xax)))
            raise
        return([batch_imgs, batch_imgs_init, batch_centers], batch_labels)
