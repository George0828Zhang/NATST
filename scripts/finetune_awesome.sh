TRAIN_FILE=/media/george/Data/mustc/en-de/alignments-distill/corpus.txt
EVAL_FILE=/media/george/Data/mustc/en-de/alignments-distill/dev_corpus.txt
OUTPUT_DIR=/media/george/Data/mustc/en-de/alignments-distill/finetune

CUDA_VISIBLE_DEVICES=0 awesome-train \
    --output_dir=$OUTPUT_DIR \
    --model_name_or_path=bert-base-multilingual-cased \
    --extraction 'softmax' \
    --do_train \
    --train_tlm \
    --train_so \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --block_size 256 \
    --should_continue \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --save_steps 4000 \
    --max_steps 20000 \
    --do_eval \
    --eval_data_file=$EVAL_FILE \
    --fp16