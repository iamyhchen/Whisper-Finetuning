python3 train/finetune_with_lora.py \
    --data_dir dataset/train-taipu-char \
    --output_dir output/taipu-char-largev2-lora \
    --model_name openai/whisper-large-v2 \
    --language zh \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 5 \
    # --resume_from_checkpoint

python3 train/get_model.py \
    --finetune_method lora \
    --output_dir output/taipu-char-largev2-lora \
    --model_checkpoint output/taipu-char-largev2-lora/checkpoint-XXX \
    --model_dir model/taipu-char-largev2-lora

python evaluate/inference.py \
  --data_dir dataset/train-taipu-char \
  --model_dir model/taipu-char-largev2-lora \
  --pred_dir predict/taipu-char-largev2-lora \