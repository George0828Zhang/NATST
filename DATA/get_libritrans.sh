#!/usr/bin/env bash
SRC=es
TGT=fr
DATA_ROOT=/groups/public/jeffeuxmartin/datasets/LibriTrans/DS91
vocab=8000
vtype=unigram

# ST
python prep_libritrans_data.py \
  --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
  --lowercase --task st
