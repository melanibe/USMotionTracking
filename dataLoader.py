import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from block_matching_utils import find_template_pixel
from tensorflow import keras

class DataLoader(keras.utils.Sequence):
    def __init__(self,
                 data_dir,
                 list_dir,
                 batch_size,
                 width_template=60,
                 resolution_df=None,
                 num_iterations=3000,
                 shuffle = True):
        self.num_iterations = num_iterations
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
                    Image.open(os.path.join(self.data_dir,
                                            subfolder, 'Data', "00001.png"))
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
                self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.orig_labels_x) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        [batch_imgs, batch_imgs_init, batch_centers], batch_labels = self.__data_generation(indexes)
        return [batch_imgs, batch_imgs_init, batch_centers], batch_labels   

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.orig_labels_x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        batch_orig_centers = np.zeros((len(indexes), 2))
        batch_centers = np.zeros((len(indexes), 2))
        batch_orig_centers[:,0] = self.orig_labels_x[indexes]
        batch_orig_centers[:,1] = self.orig_labels_y[indexes]
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
            img_init = np.asarray(Image.open(self.list_imgs_init[idx]))
            c1_init = self.list_init_x[idx]
            c2_init = self.list_init_y[idx]
            xax, yax = find_template_pixel(
                c1_init,
                c2_init,
                width=self.width_template)
            batch_imgs_init[i,:,:] = img_init[np.ravel(yax), np.ravel(xax)].reshape(self.width_template+1,
                                                                                self.width_template+1)
            # true location
            c1, c2 = self.orig_labels_x[idx], self.orig_labels_y[idx]
            # perturbed center of the template
            # N(0,10) i.e. 95% perturbation are in -20;20.
            u_x = np.random.randn()*10
            u_y = np.random.rand()*10
            c1_perturbed = c1 - u_x
            c2_perturbed = c2 - u_y
            batch_centers[i,0] = c1_perturbed
            batch_centers[i,1] = c2_perturbed
            # labels is the coord wrt to the center of
            # the pixel so here c1 = c1_perturbed - 2
            # label_x = -2 i.e. c1 = c1_perturbed + label
            batch_labels[i,0] = c1
            batch_labels[i,1] = c2
            xax, yax = find_template_pixel(
                c1_perturbed,
                c2_perturbed,
                width=self.width_template)
            batch_imgs[i] = img[np.ravel(yax), np.ravel(xax)].reshape(self.width_template+1,
                                                                        self.width_template+1)
        return([batch_imgs, batch_imgs_init, batch_centers], batch_labels)


if __name__ == "__main__":
    data_dir = os.getenv('DATA_PATH')
    print(data_dir)
    listdir = np.asarray([dI for dI in os.listdir(data_dir) if (
        os.path.isdir(os.path.join(data_dir, dI))
        and not dI == 'feats_matrices')])
    d = DataLoader(data_dir, listdir, 25)
    it = d.getBatchIterator()
    batch_imgs, batch_imgs_init, \
        batch_labels_x, batch_labels_y, \
        batch_orig_labels_x, batch_orig_labels_y = next(it)
    print(batch_labels_x)
    print(batch_orig_labels_x)
    print(batch_imgs.shape)
    print(batch_imgs_init.shape)
