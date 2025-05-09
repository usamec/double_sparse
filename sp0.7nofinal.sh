#!/bin/bash
echo "Launched at $(date)"
echo "Job ID: ${SLURM_JOBID}"
echo "Node list: ${SLURM_NODELIST}"
echo "Submit dir.: ${SLURM_SUBMIT_DIR}"
echo "Numb. of cores: ${SLURM_CPUS_PER_TASK}"
echo $SHELL

echo "Lets get this party started!"

python llama.py meta-llama/Llama-2-70b-hf c4 --sparsity 0.7 --no-final | tee logs/llama2-70-0.7-no-final;
