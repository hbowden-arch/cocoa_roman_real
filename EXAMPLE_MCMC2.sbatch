#!/bin/bash

#SBATCH --job-name=EP1
#SBATCH --output=EP1-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
##SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=7
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --account=behroozi
##SBATCH --exclusive

# Clear the environment from any previously loaded modules
#module purge > /dev/null 2>&1
#source ~/.bashrc 

#echo Running on host `hostname`
#echo Time is `date`
#echo Directory is `pwd`
#echo Slurm job NAME is $SLURM_JOB_NAME
#echo Slurm job ID is $SLURM_JOBID
#echo Number of task is $SLURM_NTASKS
#echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
source activate cocoa
source start_cocoa.sh

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

export tmp=$((${SLURM_NTASKS}-1))

$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/roman_real/EXAMPLE_MCMC2.yaml -r
