import argparse
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into Whisper base model or get baseline model")
    parser.add_argument("--finetune_method", type=str, required=True, help="baseline or lora")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to processor directory (e.g., output/whisper-finetuned)")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to model checkpoint (e.g., output/whisper-finetuned/checkpoint-XXX)")
    parser.add_argument("--model_dir", type=str, default="model/merge_model", help="Directory to save the merge model")
    parser.add_argument("--local_files_only", action="store_true", help="Only load models from local disk")

    args = parser.parse_args()

    # 驗證路徑
    assert os.path.exists(args.output_dir), f"processor dir 不存在: {args.output_dir}"
    assert os.path.exists(args.model_checkpoint), f"模型檢查點不存在: {args.model_checkpoint}"

    if args.finetune_method == "lora":
        # 讀取 PEFT 配置
        peft_config = PeftConfig.from_pretrained(args.model_checkpoint)

        # 載入 Whisper base 模型
        base_model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            local_files_only=args.local_files_only,
            device_map="auto" 
        )

        # 載入 LoRA adapter 並合併
        model = PeftModel.from_pretrained(
            base_model,
            args.model_checkpoint,
            local_files_only=args.local_files_only
        )

        model = model.merge_and_unload()
        model.eval()

        # 儲存目錄
        os.makedirs(args.model_dir, exist_ok=True)
        model.save_pretrained(args.model_dir)

        # 同步儲存 processor（tokenizer + feature_extractor）
        processor = WhisperProcessor.from_pretrained(
            peft_config.base_model_name_or_path,
            local_files_only=args.local_files_only
        )
        processor.save_pretrained(args.model_dir)


    elif args.finetune_method == "baseline":
        # 載入 Whisper base 模型
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_checkpoint,
            local_files_only=args.local_files_only,
            device_map="auto"
        )

        # 儲存目錄
        os.makedirs(args.model_dir, exist_ok=True)
        model.save_pretrained(args.model_dir)

        # 同步儲存 processor（tokenizer + feature_extractor）
        processor = WhisperProcessor.from_pretrained(
            args.output_dir,
            local_files_only=args.local_files_only
        )
        processor.save_pretrained(args.model_dir)

    print("merge completed!")

if __name__ == "__main__":
    main()