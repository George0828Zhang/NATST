#!/usr/bin/env bash
TASK=mt_distill_small
. ./data_path.sh

### st
metric=bleu
maximize=true

CONF=$DATA/config_${lang}.yaml

export CUDA_VISIBLE_DEVICES=0
FP16=true

python -m fairseq_cli.hydra_train \
    task.data=$DATA \
    task.config_yaml=$CONF \
    common.fp16=$FP16 \
    common.user_dir=.. \
    common.tensorboard_logdir=logdir/$TASK \
    dataset.train_subset=train_${lang} \
    dataset.valid_subset=dev_${lang} \
    checkpoint.save_dir=checkpoints/$TASK \
    checkpoint.best_checkpoint_metric=$metric \
    checkpoint.maximize_best_checkpoint_metric=$maximize \
    --config-dir ../config/ \
    --config-name distill