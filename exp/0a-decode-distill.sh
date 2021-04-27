#!/usr/bin/env bash
TASK=mt_distill_small
. ./data_path.sh
CONF=$DATA/config_${lang}.yaml
CHECKDIR=./checkpoints/${TASK}
CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
RESULT=./distilled

EXTRAARGS="--objectives mt --scoring sacrebleu --sacrebleu-tokenizer 13a"
GENARGS="--beam 5 --max-len-a 1.2 --max-len-b 10 --lenpen 1.1 --skip-invalid-size-inputs-valid-test"

export CUDA_VISIBLE_DEVICES=1

python -m fairseq_cli.generate ${DATA} --user-dir .. \
  --config-yaml ${CONF} --gen-subset train_${lang} \
  --task speech_to_text_multi_task \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} --max-tokens 80000 \
  --model-overrides '{"load_pretrained_encoder_from": None}' \
  --results-path ${RESULT} \
  ${GENARGS} ${EXTRAARGS}
  
