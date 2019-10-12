# A Combined Neural and Temporal Approach for Tracking Anatomical Features in Liver.

##### Mélanie Bernhardt - ETH Zurich - May 2019

#### Abstract
Ultrasound motion tracking is required for various medical applications. In this report, we describe a combined approach for tracking anatomical landmarks in liver during respiration, based on the CLUST Challenge. The proposed method combines a local Siamese-CNN and a Ridge Regression temporal model for feature localization at each frame. The method was developed and fine-tuned via 5-fold across-sequence cross-validation and then evaluated on the CLUST Challenge Test set.

#### About this repository 
This repository contains the code related to the CLUST project.
The report describing the methods and the results can also be found under `report.pdf`.


## Setup
In order to run any code of this repository 3 environment variables have to be set:

* `EXP_PATH` the path to the directory saving the checkpoints
* `DATA_PATH` the path to the training data
* `TEST_PATH` the path to the testing data.

## Main files

* To run cross_validation evaluation use `global_tracking.py`. Set the parameters you wish to use and the name of the experiment in the parameters dictionary at the end of the file and run.
* To train, predict, visualize the predictions simply use the jupyter notebook cells in `train_predict_visualize`.

## Saved models

* The weights of the final model used for the report can found under : https://drive.google.com/drive/folders/1LyS0t0LY35EOwCA1OV6cGAPEC-MKEIJx?usp=sharing
* This folder also contains the tracking results videos for each feature in the test set.

