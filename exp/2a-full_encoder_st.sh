#!/usr/bin/env bash
TASK=full_encoder_st
. ./data_path.sh
ASR_MODEL=checkpoints/ar_asr/avg_best_5_checkpoint.pt

### st
metric=bleu
maximize=true

CONF=$DATA/config_${lang}.yaml

export CUDA_VISIBLE_DEVICES=0
FP16=true

python -m fairseq_cli.hydra_train \
    model=full_encoder \
    model.load_pretrained_encoder_from=${ASR_MODEL} \
    task=st_only \
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