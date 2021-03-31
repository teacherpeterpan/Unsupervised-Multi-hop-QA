python train_stage12.py \
    --do_lower_case \
    --do_train \
    --train_file ../Data/processed/Paraphrase_New/stage1_train_data.json \
    --resource_dir ../WikiTables-WithLinks \
    --learning_rate 2e-6 \
    --option stage1 \
    --num_train_epochs 3.0 \
    --gpu_index 1 \
    --cache_dir ./tmp/

python train_stage12.py \
    --do_lower_case \
    --do_train \
    --train_file ../Data/processed/Paraphrase_New/stage2_train_data.json \
    --resource_dir ../WikiTables-WithLinks \
    --learning_rate 5e-6 \
    --option stage2 \
    --num_train_epochs 3.0 \
    --gpu_index 2 \
    --cache_dir ./tmp/

python train_stage3.py \
    --do_train  \
    --do_lower_case \
    --train_file ../Data/processed/Paraphrase_New/stage3_train_data.json \
    --resource_dir ../WikiTables-WithLinks \
    --per_gpu_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 4.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --threads 8 \
    --gpu_index 4 \
    --cache_dir ./tmp/