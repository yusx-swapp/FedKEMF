#!/bin/bash

#SBATCH --nodes=1 # request one node

#SBATCH --cpus-per-task=8  # ask for 8 cpus

#SBATCH --mem=16G # Maximum amount of memory this job will be given, try to estimate this to the best of your ability. This asks for 128 GB of ram.

#SBATCH --gres=gpu:1 #If you just need one gpu, you're done, if you need more you can change the number

#SBATCH --partition=gpu #specify the gpu partition
#asdas SBATCH --nodelist frost-3

#SBATCH --time=0-18:00:00 # ask that the job be allowed to run for 2 days, 2 hours, 30 minutes, and 2 seconds.

# everything below this line is optional, but are nice to have quality of life things

#SBATCH --output=cifar100_res32.%J.out # tell it to store the output console text to a file called job.<assigned job number>.out

#SBATCH --error=cifar100_res32.%J.err # tell it to store the error messages from the program (if it doesn't write them to normal console output) to a file called job.<assigned job muber>.err

#SBATCH --job-name="multi-head fl" # a nice readable name to give your job so you know what it is when you see it in the queue, instead of just numbers

# under this we just do what we would normally do to run the program, everything above this line is used by slurm to tell it what your job needs for resources

# let's load the modules we need to do what we're going to do

cd /work/LAS/jannesar-lab/ysx/KDFL

# let's load the modules we need to do what we're going to do
module load miniconda3/4.3.30-qdauveb
module load ml-gpu/20220603

source activate kdfl
# the commands we're running are below

nvidia-smi
ml-gpu python3 knowlege_aggregation.py --comm_round=400 --k_model='resnet20' --model='resnet32' --dataset=cifar100 --batch-size=128 --epochs=20 --n_parties=10 --sample=0.7 --logdir='./logs_cifar100/'

#ml-gpu python3 knowlege_aggregation.py --comm_round=400 --model='simple-cnn' --dataset='mnist' --k_model='simple-cnn' --lr=0.001 --batch-size=128 --epochs=10 --n_parties=10 --sample=1 --logdir='./logs_femnist/'

#--partition=noniid-labeldir

#srun --nodes 1 --tasks 1 --cpus-per-task=8 --mem=64G --partition interactive --gres=gpu:1 --partition=gpu --time 8:00:00 --pty /usr/bin/bash

#chmod +x script-name-here.sh

#sbatch slurm.sh

#scontrol show job 653783
#--gres=gpu:1 563573
#--gres=gpu:v100-pcie-16G:1

#760729