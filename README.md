# Non-autoregressive Speech Translation
- Use a single transformer encoder trained with Connectionist Temporal Classification (CTC) loss.

## Setup

1. Install fairseq
```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout cfbf0dd
pip install --upgrade . 
```
2. (Optional) [Install](docs/apex_installation.md) apex for faster mixed precision (fp16) training.
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Data Preparation
This section introduces the data preparation for training and evaluation. Following will be based on MuST-C.

1. [Download](https://ict.fbk.eu/must-c/) and unpack the package.
```bash
cd ${DATA_ROOT}
tar -zxvf MUSTC_v1.0_en-de.tar.gz
```
2. In `DATA/get_mustc.sh`, set `DATA_ROOT` to the path of speech data (the directory of previous step).
3. Preprocess data with
```bash
bash DATA/get_mustc.sh
```
The output manifest files should appear under `${DATA_ROOT}/en-de/`. 

Configure environment and path in `exp/data_path.sh` before training:
```bash
export SRC=en
export TGT=de
export DATA=/media/george/Data/mustc/${SRC}-${TGT} # should be ${DATA_ROOT}/${SRC}-${TGT}
source ~/envs/fair/bin/activate # add this to use venv located at ~/envs/fair
```

> **_NOTE:_**  subsequent commands assume the current directory is in `exp/`.
## Sequence-Level KD
We need a machine translation model as teacher for sequence-KD. The following command will train the nmt model with transcription and translation
```bash
bash 0-mt_distill.sh
```
Average the checkpoints to get a better model
```bash
CHECKDIR=checkpoints/mt_distill_small
CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
python ../scripts/average_checkpoints.py \
  --inputs ${CHECKDIR} --num-best-checkpoints 5 \
  --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
```
To distill the training set, run 
```bash
bash 0a-decode-distill.sh # generate prediction at ./distilled/train_st.tsv
bash 0b-create-distill-tsv.sh # generate distillation data at ${DATA_ROOT}/distill_${lang}.tsv
```

## ASR Pretraining
We also need an offline ASR model to initialize our ST models. For autoregressive ST, we initialize with autoregressive ASR. For other models involving causal encoder, we initialize with autoregressive ASR with a causal encoder.
```bash
bash 1-autoregressive_asr.sh # autoregressive ASR
bash 3-causal_to_causal_asr.sh # autoregressive ASR with a causal encoder
```

## Autoregressive ST
We can now train autoregressive ST model as a baseline. To do this, run
> **_NOTE:_**  to train with the distillation set, set `dataset.train_subset` to `distill_${lang}` in the script.
```bash
bash 1a-autoregressive_st.sh
```

## Non-autoregressive ST
We can train a fully non-autoregressive ST model with encoder only. To do this, run
> **_NOTE:_**  to train with the distillation set, set `dataset.train_subset` to `distill_${lang}` in the script.
```bash
bash 2a-full_encoder_st.sh
```

<!-- ## Causal Encoder ST
### Monotonic Dataset
To train a causal ST model, we first need to create a monotonic translation dataset. We can do this by the script `../scripts/create_reorder_fastalign.sh`. 

After [installing](docs/external_setup.md#fast_align) `fast-align` and `mosesdecoder`, configure the paths in the script:
```bash
# assuming that
# fairseq is install at ${FAIRSEQ_ROOT}
# fast_align at ${FASTALIGN_ROOT}
# mosesdecoder at ${MOSES_ROOT}
# this repor is at ${THIS_REPO_ROOT}
# speech data is at ${DATA_ROOT}

SCRIPT=${FAIRSEQ_ROOT}/fairseq/scripts/build_sym_alignment.py
FASTALIGN=${FASTALIGN_ROOT}/fast_align/build
MOSES=${MOSES_ROOT}
REORDER=${THIS_REPO_ROOT}/scripts/reorder.py
DISTILL=${THIS_REPO_ROOT}/DATA/create_distillation_tsv.py
# . ~/envs/apex/bin/activate # activate venv if needed

REPLACE="tgt_text"
DATA=${DATA_ROOT}
TSV=${DATA}/distill_st.tsv
ALNDIR=${DATA}/alignments-distill
OUT=${DATA}/reorder_st.tsv
SRC=en
TGT=de
```
Then run this script to create the reordered dataset.
```bash
bash ../scripts/create_reorder_fastalign.sh
```
### Training
```bash
bash 4a-causal_encoder_st.sh # by default will use the `reorder_st.tsv` created above.
``` -->

<!-- ## Causal Encoder, NAR Decoder ST
```bash
bash 5-causal_to_nat_st.sh
```

## Multitask Learning
We can also train a multitask learning model where the causal encoder is trained with transcription, and the NAR decoder with translation. Run
```bash
bash 6-causal_to_nat_mtl.sh
```

## Training with Matching Loss
Set the hyperparameters in `config/criterion/matching_criterion.yaml`:
```yaml
criterion: 
  _name: matching_criterion
  aux_factor: 1.0             # loss weight
  aux_type: mse               # can be either mse / huber / cos (cos failed, not recommended)
  use_reinforce: False        # (reinforce failed, not recommended)
  stop_grad_embeddings: True  # leave true to prevent collapse.
  zero_infinity: True
```
Train the matching model with
```bash
bash 7-matching.sh
``` -->

## Inference & Evaluation
```bash
bash ../eval.sh
```

## More Info
See [more](docs/more_info.md) for hyperparameter tuning, config setup and model relations.