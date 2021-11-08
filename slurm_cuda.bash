#!/bin/bash
#SBATCH -J dnam     # job name
#SBATCH -o log_slurm.o%j  # output and error file name (%j expands to jobID)
#SBATCH -n 1             # total number of tasks requested
#SBATCH -N 1 		  # number of nodes you want to run on	
#SBATCH -p nam-bio # queue (partition) -- defq, eduq, gpuq, shortq, nam(Our group), bsudfq(for all BSU)
#SBATCH -t 36:00:00       # run time (hh:mm:ss) - 12.0 hours in this example.
#SBATCH --gres=gpu:1
#SBATCH --mail-user=golammdmortuza@boisestate.edu
# Remove previous log file
module load slurm
module load cuda10.0

# python generate_data.py 1
python main.py 1
