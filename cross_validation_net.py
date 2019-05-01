from dataLoader import DataLoader, compute_euclidean_distance
import numpy as np
import os
from custom_KFold import MyKFold
from network import create_model
import pandas as pd


def run_CV(params_dict, data_dir, n_splits=5):
    width = params_dict.get('width') if params_dict.get(
        'width') is not None else 60
    n_epochs = params_dict.get('n_epochs') if params_dict.get(
        'n_epochs') is not None else 15
    h1 = params_dict.get('h1') if params_dict.get('h1') is not None else 32
    h2 = params_dict.get('h2') if params_dict.get('h2') is not None else 64
    h3 = params_dict.get('h3') if params_dict.get('h3') is not None else 128
    embed_size = params_dict.get('embed_size') if params_dict.get(
        'embed_size') is not None else 128
    dropout_rate = params_dict.get('dropout_rate') if params_dict.get(
        'dropout_rate') is not None else 0
    use_batchnorm = params_dict.get('batch_norm') if params_dict.get(
        'batch_norm') is not None else True
    # KFold iterator
    kf = MyKFold(data_dir, n_splits=n_splits)
    fold_iterator = kf.getFolderIterator()
    eucl_dist_dict = {}
    eucl_dist_per_fold = []
    mse_per_fold = []
    for traindirs, testdirs in fold_iterator:
        # Generators
        training_generator = DataLoader(
            data_dir, traindirs, 32, width_template=width)
        validation_generator = DataLoader(
            data_dir, testdirs, 32, width_template=width, type='val')
        # Design model
        model = create_model(width+1, h1, h2, h3,
                             embed_size=embed_size,
                             drop_out_rate=dropout_rate,
                             use_batch_norm=use_batchnorm)
        # Train model on dataset
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            epochs=n_epochs,
                            workers=4)
        list_dist_curr_fold = []
        mse_curr_fold = []
        for testfolder in testdirs:
            eucl_dist = 0
            batch_counter = 0
            gen = DataLoader(data_dir, [testfolder],
                             1, width_template=width, type='val')
            mse_curr_fold.append(model.evaluate_generator(gen))
            n = gen.__len__()
            print(n)
            for data, labels in gen:
                if batch_counter < n:
                    batch_counter += 1
                    preds = model.predict(x=data)
                    eucl_dist += compute_euclidean_distance(
                        gen.resolution_df, testfolder, preds, labels)
                else:
                    break
            print(eucl_dist/batch_counter)
            eucl_dist_dict[testfolder] = eucl_dist/batch_counter
            list_dist_curr_fold.append(eucl_dist/batch_counter)
        eucl_dist_per_fold.append(np.mean(list_dist_curr_fold))
        mse_per_fold.append(np.mean(mse_curr_fold))
    print(np.mean(eucl_dist_per_fold))
    return eucl_dist_per_fold, eucl_dist_dict
    # labels = model.predict(validation_generator)


if __name__=="__main__":
    np.random.seed(seed=42)
    # Get the training data
    data_dir = os.getenv('DATA_PATH')
    print(data_dir)
    # default experiment
    params = {'dropout_rate': 0.5, 'n_epochs': 80,
            'h3': 0, 'embed_size': 512, 'width': 60}
    eucl_dist_per_fold, eucl_dist_dict = run_CV(params, data_dir)
    save_res = pd.DataFrame(eucl_dist_dict, index=[0])
    params_df = pd.DataFrame({}, index=[0])
    save_res['mean_over_folds'] = np.mean(eucl_dist_per_fold)
    save_res['std_over_folds'] = np.std(eucl_dist_per_fold)
    result = pd.concat([save_res, params_df], axis=1)
    result.to_csv(os.path.join('data_dir', 'results', 'default.csv'), index=False)
    print(result)
