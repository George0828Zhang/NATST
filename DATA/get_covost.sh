#!/usr/bin/env bash
SRC=es
TGT=en
DATA_ROOT=/media/george/Data/covost
vocab=8000
vtype=unigram

# ST
python prep_covost_data.py \
  --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
  --src-lang $SRC --tgt-lang $TGT --lowercase --rm-quote