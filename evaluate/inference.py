import os
import pandas as pd
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import evaluate
import argparse

# ---------- 參數與路徑 ----------
argparser = argparse.ArgumentParser(description="Whisper Merged Model Inference")
argparser.add_argument("--data_dir", type=str, required=True, help="Directory containing the test dataset")
argparser.add_argument("--model_dir", type=str, required=True, help="Directory containing the merged model")
argparser.add_argument("--pred_dir", type=str, default=None, help="Directory to save predictions")
args = argparser.parse_args()

TEXT_FILE = os.path.join(args.data_dir, "test/text")
AUDIO_FILE = os.path.join(args.data_dir, "test/audio_paths")
MODEL_DIR = args.model_dir

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

# ---------- 載入模型與處理器 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ---------- 載入與處理資料 ----------
dataset = load_data(TEXT_FILE, AUDIO_FILE)

def map_to_prediction(batch):
    inputs = processor(batch["path"]["array"], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad(): 
        predicted_ids = model.generate(input_features)
    batch["predicted_text"] = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return batch

result_dataset = dataset.map(map_to_prediction, remove_columns=["path"], desc="Running inference", num_proc=1)

# ---------- 評估指標 ----------
wer = evaluate.load("wer")
cer = evaluate.load("cer")

references = result_dataset["text"]
predictions = result_dataset["predicted_text"]

print("=== Evaluation Results ===")
print(f"WER: {wer.compute(predictions=predictions, references=references) * 100:.2f}%")
print(f"CER: {cer.compute(predictions=predictions, references=references) * 100:.2f}%")

# ---------- 寫入預測結果 ----------
os.makedirs(args.pred_dir, exist_ok=True)
output_path = os.path.join(args.pred_dir, "predict.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"WER: {wer.compute(predictions=predictions, references=references) * 100:.2f}%\n")
    f.write(f"CER: {cer.compute(predictions=predictions, references=references) * 100:.2f}%\n")
    f.write("=== Predictions ===\n")
    for ref, pred in zip(references, predictions):
        f.write(f"[GT]   {ref}\n")
        f.write(f"[Pred] {pred}\n")
        f.write("---\n")

print(f"\n預測結果已寫入: {output_path}")
