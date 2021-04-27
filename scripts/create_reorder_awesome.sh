#!/usr/bin/env bash
SCRIPT="-m awesome_align.run_align"
AWESOMEMDL=/media/george/Data/mustc/en-de/alignments-distill/finetune
MOSES=/home/george/utility/mosesdecoder
REORDER=/home/george/Projects/stream-st/scripts/reorder.py
DISTILL=/home/george/Projects/stream-st/DATA/create_distillation_tsv.py
# . ~/envs/apex/bin/activate

REPLACE="tgt_text"
DATA=/media/george/Data/mustc/en-de #/livingrooms/george/mustc/en-de
TSV=${DATA}/distill_st.tsv
ALNDIR=${DATA}/alignments-distill
OUT=${DATA}/reorder2_st.tsv
SRC=en
TGT=de
PREFIX=$(basename ${TSV%.tsv})

mkdir -p ${ALNDIR}

echo "extracting language pairs from ${TSV}"
TOKENIZE=${MOSES}/scripts/tokenizer/tokenizer.perl
SRCFILE=${ALNDIR}/${PREFIX}.${SRC}
TGTFILE=${ALNDIR}/${PREFIX}.${TGT}
if [ -f "${SRCFILE}" ]; then
	echo "${SRCFILE} exists, skipping tokenize"
else
	tail -n +2 ${TSV} | cut -f4 | ${TOKENIZE} -q -l ${SRC} > ${SRCFILE}
fi
if [ -f "${TGTFILE}" ]; then
	echo "${TGTFILE} exists, skipping tokenize"
else
	tail -n +2 ${TSV} | cut -f5 | ${TOKENIZE} -q -l ${TGT} > ${TGTFILE}
fi

echo "aligning ..."
CORPUS=${ALNDIR}/${PREFIX}.${SRC}-${TGT}
ALIGNOUT=${ALNDIR}/aligned.awesome
if [ -f "${ALIGNOUT}" ]; then
	echo "${ALIGNOUT} exists, skipping alignment"
else
	paste ${SRCFILE} ${TGTFILE} | sed "s/\t/ ||| /" > ${CORPUS}
	python ${SCRIPT} --output_file=${ALIGNOUT} \
    --model_name_or_path=${AWESOMEMDL} \
    --data_file=${CORPUS} \
    --extraction 'softmax' \
    --batch_size 32
fi    

echo "reordering ..."
if [ -f "${TGTFILE}.reord" ]; then
	echo "${TGTFILE}.reord exists, skipping reordering"
else	
	python ${REORDER} -s ${SRCFILE} -t ${TGTFILE} -a ${ALIGNOUT} -o ${TGTFILE}.reord
fi

# make compatible to create-distill.py
DETOKENIZE=${MOSES}/scripts/tokenizer/detokenizer.perl
if [ -f "${TGTFILE}.ready" ]; then
	echo "${TGTFILE}.ready exists, skipping format convert"
else	
	echo "converting to format accepted by create_distillation_tsv"
	cat ${TGTFILE}.reord | ${DETOKENIZE} -q -l ${TGT} | nl -v0 -w1 | sed -r "s/^([0-9]+)/D-\1\t0.0/" > ${TGTFILE}.ready
fi


if [ -f "${OUT}" ]; then
	echo "${OUT} exists, skipping tsv creation"
else	
	echo "creating new data ..."
	python ${DISTILL} \
		--train-file ${TSV} \
		--distill-file ${TGTFILE}.ready \
		--replace-col ${REPLACE} \
		--out-file ${OUT}
fi