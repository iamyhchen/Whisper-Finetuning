python3 train/baseline.py \
    --data_dir dataset/train-taipu-char \
    --output_dir output/taipu-char-medium \
    --model_name openai/whisper-medium \
    --language zh \
    --num_epoch 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    # --resume_from_checkpoint

python3 train/get_model.py \
    --finetune_method baseline \
    --output_dir output/taipu-char-medium \
    --model_checkpoint output/taipu-char-medium/checkpoint-XXX \
    --model_dir model/taipu-char-medium

python evaluate/inference.py \
  --data_dir dataset/taipu-char-medium \
  --model_dir model/taipu-char-medium \
  --pred_dir predict/taipu-char-medium \