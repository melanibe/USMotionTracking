import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from block_matching_utils import find_template_pixel


class DataLoader:
    def __init__(self,
                 data_dir,
                 list_dir,
                 batch_size,
                 width_template=60,
                 resolution_df=None,
                 num_iterations=3000):
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

    def getBatchIterator(self):
        n = len(self.orig_labels_x)
        num_batch_per_epoch = n//self.batch_size
        n_epochs = self.num_iterations//num_batch_per_epoch+1
        for _ in range(n_epochs):
            shuffle_indices = np.random.permutation(np.arange(n))
            shuffle_orig_labels_x = self.orig_labels_x[shuffle_indices]
            shuffle_orig_labels_y = self.orig_labels_y[shuffle_indices]
            shuffle_init_x = self.list_init_x[shuffle_indices]
            shuffle_init_y = self.list_init_y[shuffle_indices]
            shuffled_imgs = self.list_imgs[shuffle_indices]
            shuffled_imgs_init = self.list_imgs_init[shuffle_indices]
            for batch_num in range(num_batch_per_epoch):
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, n)
                batch_orig_labels_x = shuffle_orig_labels_x[start_index:end_index]
                batch_orig_labels_y = shuffle_orig_labels_y[start_index:end_index]
                batch_init_x = shuffle_init_x[start_index:end_index]
                batch_init_y = shuffle_init_y[start_index:end_index]
                batch_labels_x = np.zeros(len(range(start_index, end_index)))
                batch_labels_y = np.zeros(len(range(start_index, end_index)))
                batch_imgs = np.zeros(
                    (len(range(start_index, end_index)),
                     self.width_template+1,
                     self.width_template+1)
                )
                batch_imgs_init = np.zeros(
                    (len(range(start_index, end_index)),
                     self.width_template+1,
                     self.width_template+1)
                )
                for i in range(start_index, end_index):
                    img = np.asarray(Image.open(shuffled_imgs[i]))
                    img_init = np.asarray(Image.open(shuffled_imgs_init[i]))
                    c1_init = shuffle_init_x[i]
                    c2_init = shuffle_init_y[i]
                    xax, yax = find_template_pixel(
                        c1_init,
                        c2_init,
                        width=self.width_template)
                    batch_imgs_init[i] = img_init[np.ravel(yax), np.ravel(xax)].reshape(self.width_template+1,
                                                                                        self.width_template+1)
                    # true location
                    c1, c2 = shuffle_orig_labels_x[i], shuffle_orig_labels_y[i]
                    # perturbed center of the template
                    u_x = np.random.randn()*10 # N(0,10) i.e. 95% perturbation are in -20;20.
                    u_y = np.random.rand()*10
                    c1_perturbed = c1 - u_x 
                    c2_perturbed = c2 - u_y
                    # labels is the coord wrt to the center of
                    # the pixel so here c1 = c1_perturbed - 2
                    # label_x = -2 i.e. c1 = c1_perturbed + label
                    batch_labels_x[i] = u_x
                    batch_labels_y[i] = u_y
                    xax, yax = find_template_pixel(
                        c1_perturbed,
                        c2_perturbed,
                        width=self.width_template)
                    batch_imgs[i] = img[np.ravel(yax), np.ravel(xax)].reshape(self.width_template+1,
                                                                              self.width_template+1)

                yield batch_imgs, batch_imgs_init, batch_labels_x, batch_labels_y, batch_orig_labels_x, batch_orig_labels_y


if __name__ == "__main__":
    data_dir = os.getenv('DATA_PATH')
    print(data_dir)
    d = DataLoader(data_dir, ['ETH-02-1', 'ETH-01-1'], 25)
    it = d.getBatchIterator()
    batch_imgs, batch_imgs_init, \
        batch_labels_x, batch_labels_y, \
        batch_orig_labels_x, batch_orig_labels_y = next(it)
    print(batch_labels_x)
    print(batch_orig_labels_x)
    print(batch_imgs.shape)
    print(batch_imgs_init.shape)
