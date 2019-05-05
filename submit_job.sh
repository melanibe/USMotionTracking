#!/bin/bash
## SET THE FOLLOWING VARIABLES ACCORDING TO YOUR SYSTEM ##
CUDA_HOME=/scratch_net/biwidl202/melanibe/apps/cuda-9.0


# therwise the detaul shell would be used
#$ -S /bin/bash

## Pass env. vars of the workstation to the GPU node
##$-V

## <= 1h is short queue, <= 6h is middle queue, <= 48h is long queue
#$ -q gpu.middle.q@*

## The maximum memory usage of this job (below 4G does not make much sense)
#$ -l gpu
#$ -l h_vmem=40G



## stderr and stdout are merged together to stdout
##$ -j y
#
# logging directory, preferably on your scratch
##$ -o /scratch_net/biwidl202/melanibe/logs
#


# cuda paths
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH



# call your calculation exec.
export DATA_PATH='/scratch_net/biwidl202_second/melanibe/CLUST/CLUST_Data'
export EXP_PATH='/scratch_net/biwidl202_second/melanibe/CLUST/CLUST_Runs'
source /scratch_net/biwidl202/melanibe/anaconda3/bin/activate
CUDA_VISIBLE_DEVICES=$SGE_GPU python global_tracking.py


