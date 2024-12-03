#!/bin/bash

#SBATCH --account=behroozi
#SBATCH --job-name=dv_chain
#SBATCH --output=test_dv-%A_%a.out
#SBATCH -e test_dv-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
##SBATCH --partition=standard
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_behroozi
#SBATCH -t 12:00:00

##SBATCH --job-name=EXP1
##SBATCH --output=EXP1-%j.out
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=8
##SBATCH --cpus-per-task=10
##SBATCH --partition=high_priority
##SBATCH --qos=user_qos_behroozi
##SBATCH --account=behroozi
##SBATCH --time=12:00:00

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

module purge > /dev/null 2>&1
cd /groups/behroozi/hbowden/cocoa/Cocoa

source activate cocoa
source start_cocoa.sh

##module load gsl
  
export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi


echo calculating dvs

#mpirun \
#-n ${SLURM_NTASKS} \
#--mca btl tcp,self \
#--bind-to core \
#--map-by numa:pe=${OMP_NUM_THREADS} \
#python projects/roman_real/calc_dvs.py \
#projects/roman_real/test_roman.yaml


mpirun -n 1 --oversubscribe \
--mca btl tcp,self \
--bind-to core:overload-allowed \
--rank-by core \
--map-by numa:pe=1 \
python projects/roman_real/calc_dvs.py \
projects/roman_real/test_roman.yaml

#python projects/roman_real/EXAMPLE_PROFILE1.py
#python projects/roman_real/calc_dvs.py projects/roman_real/test_roman.yaml
#python calc_dvs.py J_config_example.yaml


#mpirun -n ${SLURM_NTASKS} --mca orte_base_help_aggregate 0 --oversubscribe --report-bindings --mca btl tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} python projects/roman_real/calc_dvs.py projects/roman_real/test_roman.yaml

#mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} python projects/roman_real/calc_dvs.py projects/roman_real/test_roman.yaml

##$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} --mca btl tcp,self --bind-to core --map-by numa:pe=${OMP_NUM_THREADS} python projects/roman_real/calc_dvs.py projects/roman_real/J_config_example.yaml