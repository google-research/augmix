#!/bin/bash
#SBATCH --time=05:10:00
#SBATCH --nodes=1
#SBATCH --mem 20G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%1
#SBATCH --output=slurm/training_%A_%a.out
#SBATCH --error=slurm/training_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-siegen.de

echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

#networks="wrn resnext allconv densenet"
networks=("wrn" "resnext" "allconv" "densenet")
lr="0.1"

#p1 is the element of the array found with ARRAY_ID mod P_ARRAY_LENGTH
net=${networks[$SLURM_ARRAY_TASK_ID]}
if [[ $net = 'allconv' ]]
then
    lr='0.01'
else
    lr='0.1'
fi
python augmix_refactored/script/cifar.py --model ${net} --learning-rate ${lr}
#echo ${net} ${lr} $SLURM_ARRAY_TASK_ID

#for net in $networks
#do
    #if [[ $net = 'allconv' ]]
    #then
        #lr='0.01'
    #else
        #lr='0.1'
    #fi
    #python augmix_refactored/script/cifar.py --model ${net} --learning-rate ${lr}
#done
end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime