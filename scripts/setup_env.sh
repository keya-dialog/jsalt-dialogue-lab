#!/bin/bash

if [[ $USER = oplatek ]] ; then
  # logging to wandb.ai is as easy as setup your ENTITY and name your PROJECT
  # change it to yours probably new branch for your username
  # important commands to be run from your working directory are:
  #  wandb offline: display and log only locally do not push results to wandb.ai cloud
  #  wandb online: start/stop logging to wandb.ai cloud
  #  wandb disabled and wandb disabled : turn on and off the wandb logging completely
  export WANDB_ENTITY=metric
  export WANDB_PROJECT=llm_finetune_multiwoz22.sh
fi

if [[ $HOSTNAME = sol2 ]] ; then
  # submit using slurm on UFAL MFF CUNI slurm cluster.
  # gpu-python is drop-in replacement for python so it blocks the terminal by default
  #  it is submission script for slurm
  #  see https://github.com/oplatek/shellgit-ufal/blob/master/bin/gpu-python
  export PYTHON="gpu-python --gpu-mem 40"
elif [[ $HOSTNAME = tdll-*gpu* ]] ; then
  # We detected you are are already on node with a gpu on UFAL cluster so using slurm is not necessary.
  export PYTHON=python
else
  # By default simply use python and assume you have GPU with enough memory available.
  export PYTHON=python
fi
