import os
import pandas as pd
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate
import argparse

argparser = argparse.ArgumentParser(description="Whisper Fine-tuning Script")
argparser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset files")
argparser.add_argument("--output_dir", type=str, default="./output/whisper-finetuned", help="Directory to save the fine-tuned model")
argparser.add_argument("--model_name", type=str, default="openai/whisper-medium", help="Pre-trained Whisper model name")
argparser.add_argument("--language", type=str, default="en", help="which language to use for the model")
argparser.add_argument("--num_epoch", type=int, default=5, help="training epochs")
argparser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training")
argparser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size per device during evaluation")
argparser.add_argument("--resume_from_checkpoint", action='store_true', help="Resume training from the last checkpoint")
args = argparser.parse_args()

# ---------- 參數設定 ----------
TEXT_FILE = os.path.join(args.data_dir, "train/text")
AUDIO_FILE = os.path.join(args.data_dir, "train/audio_paths")
EVAL_TEXT_FILE = os.path.join(args.data_dir, "eval/text")
EVAL_AUDIO_FILE = os.path.join(args.data_dir, "eval/audio_paths")
MODEL_NAME = args.model_name
OUTPUT_DIR = args.output_dir
LANGUAGE = args.language
TASK = "transcribe"
NUM_EPOCHS = args.num_epoch
TRAIN_BATCH_SIZE = args.per_device_train_batch_size
EVAL_BATCH_SIZE = args.per_device_eval_batch_size
LEARNING_RATE = 1e-5

# ---------- 讀取文字與音訊路徑 ----------
def load_data(text_file, audio_file):
    text_data = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                uid, text = parts
                text_data.append((uid, text))

    audio_data = []
    with open(audio_file, "r", encoding="utf-8") as f:
        for line in f:
            uid, path = line.strip().split(maxsplit=1)
            audio_data.append((uid, path))

    text_df = pd.DataFrame(text_data, columns=["id", "text"])
    audio_df = pd.DataFrame(audio_data, columns=["id", "path"])
    df = pd.merge(text_df, audio_df, on="id")

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
    return dataset

# ---------- 前處理器 ----------
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

def prepare_example(batch):
    audio = batch["path"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# ---------- 載入模型 ----------
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# ---------- Collator ----------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
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
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    print("\n")
    print("label_ids:", label_ids[0][0:50])
    print("pred_ids :", pred_ids[0][0:50])
    print("label_str:", label_str[0])
    print("pred_str :", pred_str[0])
    print("wer:", wer)
    print("cer:", cer)
    print("\n")
    return {"wer": wer, "cer": cer}

# ---------- 載入資料並處理 ----------
train_dataset = load_data(TEXT_FILE, AUDIO_FILE)
train_dataset = train_dataset.map(prepare_example)

# Load and preprocess evaluation dataset
eval_dataset = load_data(EVAL_TEXT_FILE, EVAL_AUDIO_FILE)
eval_dataset = eval_dataset.map(prepare_example)

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
    eval_dataset=eval_dataset,  # Add evaluation dataset
    tokenizer=processor.feature_extractor,  # Use feature extractor directly
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# ---------- 儲存模型 ----------
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
