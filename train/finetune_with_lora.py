import os
import pandas as pd
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate
import argparse

# ---------- 參數與路徑設定 ----------
argparser = argparse.ArgumentParser(description="Whisper Fine-tuning Script with LoRA")
argparser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset files")
argparser.add_argument("--output_dir", type=str, default="./output/whisper-lora-finetuned", help="Directory to save the LoRA fine-tuned model")
argparser.add_argument("--model_name", type=str, help="whisper model name, e.g., openai/whisper-large-v2")
argparser.add_argument("--language", type=str, help="Language for the Whisper model")
argparser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size for training")
argparser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size for evaluation")
argparser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")
argparser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
argparser.add_argument("--resume_from_checkpoint", action='store_true', help="Resume training from the last checkpoint")

args = argparser.parse_args()

TEXT_FILE = os.path.join(args.data_dir, "train/text")
AUDIO_FILE = os.path.join(args.data_dir, "train/audio_paths")
EVAL_TEXT_FILE = os.path.join(args.data_dir, "eval/text")
EVAL_AUDIO_FILE = os.path.join(args.data_dir, "eval/audio_paths")
MODEL_NAME = args.model_name
OUTPUT_DIR = args.output_dir
LANGUAGE = args.language
TASK = "transcribe"
NUM_EPOCHS = args.num_train_epochs
TRAIN_BATCH_SIZE = args.per_device_train_batch_size
EVAL_BATCH_SIZE = args.per_device_eval_batch_size
LEARNING_RATE = args.learning_rate

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 載入資料 ----------
def load_data(text_file, audio_file):
    text_data, audio_data = [], []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                text_data.append(parts)

    with open(audio_file, "r", encoding="utf-8") as f:
        for line in f:
            audio_data.append(line.strip().split(maxsplit=1))

    df = pd.merge(pd.DataFrame(text_data, columns=["id", "text"]),
                  pd.DataFrame(audio_data, columns=["id", "path"]),
                  on="id")
    dataset = Dataset.from_pandas(df)
    return dataset.cast_column("path", Audio(sampling_rate=16000))

# ---------- 前處理器 ----------
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

def prepare_example(batch):
    audio = batch["path"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# ---------- 載入並套用 LoRA ----------
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, device_map = "auto")
base_model.config.forced_decoder_ids = None
base_model.config.suppress_tokens = []

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
    # target_modules = ["q_proj", "v_proj"]
)

base_model.enable_input_require_grads()
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# ---------- Collator ----------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ---------- 評估指標 ----------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {
        "wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": 100 * cer_metric.compute(predictions=pred_str, references=label_str),
    }

# ---------- 載入並處理資料 ----------
train_dataset = load_data(TEXT_FILE, AUDIO_FILE).map(prepare_example)
eval_dataset = load_data(EVAL_TEXT_FILE, EVAL_AUDIO_FILE).map(prepare_example)

# ---------- 訓練參數 ----------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    num_train_epochs=NUM_EPOCHS,
    gradient_checkpointing=True,
    logging_steps=100,
    save_strategy="epoch",
    do_eval=False,
    fp16=True,
    push_to_hub=False
)

# ---------- 開始訓練 ----------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# ---------- 儲存processor（tokenizer + feature_extractor）----------
processor.save_pretrained(OUTPUT_DIR)
