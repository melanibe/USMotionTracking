from dataLoader import DataLoader, compute_euclidean_distance
from network import create_model
import os
import numpy as np
from custom_KFold import MyKFold
from block_matching_utils import find_template_pixel, global_template_search
from PIL import Image
import pandas as pd
from tensorflow import keras

np.random.seed(seed=42)
# Get the training data
data_dir = os.getenv('DATA_PATH')
print(data_dir)

# default experiment
params_dict = {'dropout_rate': 0.4, 'n_epochs': 2, 'h3': 0, 'h4': 0, 'embed_size': 128}
width = params_dict.get('width') if params_dict.get(
    'width') is not None else 60
n_epochs = params_dict.get('n_epochs') if params_dict.get(
    'n_epochs') is not None else 15
h1 = params_dict.get('h1') if params_dict.get('h1') is not None else 32
h2 = params_dict.get('h2') if params_dict.get('h2') is not None else 64
h3 = params_dict.get('h3') if params_dict.get('h3') is not None else 128
h4 = params_dict.get('h4') if params_dict.get('h4') is not None else 0
embed_size = params_dict.get('embed_size') if params_dict.get(
    'embed_size') is not None else 128
dropout_rate = params_dict.get('dropout_rate') if params_dict.get(
    'dropout_rate') is not None else 0
use_batchnorm = params_dict.get('batch_norm') if params_dict.get(
    'batch_norm') is not None else True

# KFold iterator
kf = MyKFold(data_dir, n_splits=5)
fold_iterator = kf.getFolderIterator()
eucl_dist_dict = {}
eucl_dist_per_fold = []
mse_per_fold = []
for traindirs, testdirs in fold_iterator:
    # TRAIN LOCAL PREDICTION MODEL
    # Generators
    training_generator = DataLoader(
        data_dir, traindirs, 32, width_template=width)
    validation_generator = DataLoader(
        data_dir, testdirs, 32, width_template=width, type='val')
    
    # Design model
    model = create_model(width+1, h1, h2, h3, h4,
                         embed_size=embed_size,
                         drop_out_rate=dropout_rate,
                         use_batch_norm=use_batchnorm)
    # Train model on training dataset
    """
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        epochs=n_epochs,
                        steps_per_epoch=50,
                        workers=4)
    """
    #model.save_weigths(os.path.join(data_dir, 'model.h5'))

    model.load_weights(os.path.join(data_dir, 'model.h5'))
    # PREDICT WITH GLOBAL MATCHING + LOCAL MODEL ON TEST SET
    for testfolder in testdirs:
        annotation_dir = os.path.join(data_dir, testfolder, 'Annotation')
        img_dir = os.path.join(data_dir, testfolder, 'Data')
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
        list_centers = [[]]
        global_centers = [[]]
        for label_file in list_label_files:
            img_current = np.asarray(Image.open(list_imgs[0]))
            img_prev_good = np.asarray(Image.open(list_imgs[0]))
            df = pd.read_csv(label_file,
                             header=None,
                             names=['id', 'x', 'y'],
                             sep='\s+')
            c1_init, c2_init = df.loc[df['id'] == 1, ['x', 'y']].values[0, :]
            c1, c2 = df.loc[df['id'] == 1, ['x', 'y']].values[0, :]
            c1_prev, c2_prev = df.loc[df['id'] == 1, ['x', 'y']].values[0, :]
            xax, yax = find_template_pixel(c1_init, c2_init,
                                           width=width)
            template_init = img_init[np.ravel(yax), np.ravel(xax)].reshape(1, len(yax), len(xax))
            n_obs = len(list_imgs)
            print('list_imgs length {}'.format(n_obs))
            #batch_imgs = np.zeros((n_obs-1, width+1, width+1))
            #batch_imgs_init = np.tile(init_template.reshape(width+1, width+1), n_obs-1)
            for i in range(1, len(list_imgs)):
                if i%100==0:
                    print(i)
                img_prev = img_current
                # modify like in DataLoader
                img_current = np.asarray(Image.open(list_imgs[i]))
                c1, c2, maxNCC, im_prev_good, c1_prev, c2_prev = global_template_search(c1,
                                                                                          c2,
                                                                                          c1_prev,
                                                                                          c2_prev,
                                                                                          c1_init,
                                                                                          c2_init,
                                                                                          img_prev,
                                                                                          img_current,
                                                                                          img_init,
                                                                                          img_prev_good,
                                                                                          threshold_good=0.95,
                                                                                          threshold_bad=0.55,
                                                                                          width=60)
                xax, yax = find_template_pixel(c1, c2,
                                               width=width)
                template_current = img_current[np.ravel(yax), np.ravel(xax)].reshape(1, len(yax), len(xax))
                current_centers = np.asarray([c1, c2]).reshape(1,2)
                pred = model.predict(x=[template_current, template_init, current_centers])
                global_centers = np.append(global_centers, [c1, c2])
                c1, c2 = pred[0, 0], pred[0, 1]
                # In case the "good image" was updated
                if np.all(img_current==im_prev_good):
                    c1_prev, c2_prev = pred[0, 0], pred[0, 1]
                list_centers = np.append(list_centers, [c1, c2])
                if i in df.id.values:
                    diff_x = np.abs(c1-df.loc[df['id']==i, 'x'].values)
                    diff_y = np.abs(c2-df.loc[df['id']==i, 'y'].values)
                    dist = np.sqrt(diff_x**2+diff_y**2)
                    #print('Abs diff in x {}, in y {}'.format(diff_x, diff_y))
                    print('ID {} : dist diff {}'.format(i, dist))
                    if dist > 3:
                        print('Bad dist - maxNCC was {}'.format(maxNCC))
            idx = df.id.values[1:len(df)]
            df_preds = list_centers[idx-1]
            df_true = df[['x', 'y']].values[1:len(df)]
            absolute_diff = np.mean(np.abs(df_preds-df_true))
            pix_dist = np.mean(np.sqrt((df_preds[0]-df_true[0])**2+(df_preds[1]-df_true[1])**2))
            dist = compute_euclidean_distance(kf.resolution_df, testfolder, df_preds, df_true)
            print('Pixel distance folder {} is {}'.format(testfolder, pix_dist))
            print('Euclidean distance in mm {}'.format(dist))
            print('Mean absolute difference {}'.format(absolute_diff))