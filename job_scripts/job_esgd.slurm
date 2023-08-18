#!/bin/bash

# Partition for the job:
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name=esgd

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# The amount of memory in megabytes per node:
#SBATCH --mem=16384

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-11:00:00

#SBATCH --output=./outputs/slurm_logs/slurm_%A_%a.out
#SBATCH --error=./outputs/slurm_logs/slurm_%A_%a.err
#SBATCH --array=1-10

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load python/3.9.5


# Activate existing virtual environment
source /path/to/virtualenv/bin/activate

########## Script ###########

# Create an array of RNGSEEDS using job array task id and ensure they don't overlap.
MAXNUMWORKERS=8
RNGSEEDS=()
for i in $(seq 1 8); do 
    RNGSEEDS+=($(($i + ${SLURM_ARRAY_TASK_ID} * 100)));
done
echo "${RNGSEEDS[@]}"

NUMEPOCH=200
BATCHSIZE=512
NUMTRAININGDATA=60000

OPTIMIZERTYPE="entropy-sgd"
LEARNINGRATE=0.0001
SGLDGAMMA=100
SGLDNUMITER=400
SGLDNUMCHAINS=4

OUTPUTDIR="./outputs/expt_sgd"

cp -r ./data/ /tmp/data/

python main.py --save_plot --save_result --seeds "${RNGSEEDS[@]}" --optimizer ${OPTIMIZERTYPE} --outputdir "${OUTPUTDIR}" --lr ${LEARNINGRATE} --max_workers ${MAXNUMWORKERS} --batch_size ${BATCHSIZE} --num_training_data ${NUMTRAININGDATA} --sgld_gamma ${SGLDGAMMA} --sgld_num_iter ${SGLDNUMITER} --sgld_num_chains ${SGLDNUMCHAINS} --data_rootdir /tmp/data/ --epochs ${NUMEPOCH} --save_model

########## Script End ###########

# deactivate virtualenv
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
