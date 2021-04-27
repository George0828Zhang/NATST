#!/usr/bin/env bash
TASK=full_encoder_asr
. ./data_path.sh

### asr
metric=wer
maximize=false

CONF=$DATA/config_${lang}.yaml

export CUDA_VISIBLE_DEVICES=0
FP16=true

python -m fairseq_cli.hydra_train \
    model=full_encoder \
    task=asr_only \
    criterion=mtl_ctc \
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
    --config-name general