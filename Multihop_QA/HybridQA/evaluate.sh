CUDA_VISIBLE_DEVICES=0 

python train_stage12.py \
    --stage1_model stage1/2020_11_10_13_30_37/checkpoint-epoch1 --stage2_model stage2/2020_11_10_13_30_58/checkpoint-epoch0 \
    --do_lower_case \
    --resource_dir ../../Data/HybridQA/WikiTables-WithLinks \
    --predict_file ./data/human/dev_inputs.json \
    --do_eval \
    --option stage12

python train_stage3.py \
    --model_name_or_path stage3/2020_11_10_13_31_17/checkpoint-epoch3 \
    --do_stage3 \
    --do_lower_case \
    --resource_dir ../../Data/HybridQA/WikiTables-WithLinks \
    --predict_file predictions.intermediate.json \
    --per_gpu_train_batch_size 12 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --threads 8

python evaluate_script.py \
    predictions.json \
    ../../Data/HybridQA/dev_reference.json
