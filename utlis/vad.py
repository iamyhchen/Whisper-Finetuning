import os
import torch
import torchaudio
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# ---------- 設定 ---------- #
input_root = "Corpus"
output_root = "VAD_Corpus"
sampling_rate = 16000

# 載入 VAD 模型
model = load_silero_vad()

# ---------- 遍歷所有音檔 ---------- #
for root, dirs, files in os.walk(input_root):
    for file in files:
        if not file.endswith(".wav"):
            continue

        input_path = os.path.join(root, file)
        wav = read_audio(input_path, sampling_rate=sampling_rate)

        # 執行 VAD 擷取有聲段
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=sampling_rate,
            threshold=0.25,
            min_silence_duration_ms=100,
            min_speech_duration_ms=150,
            speech_pad_ms=10,
            window_size_samples=512
        )

        print(f"處理檔案：{input_path}，偵測到 {len(speech_timestamps)} 個語音段")

        if not speech_timestamps:
            print("沒有偵測到語音，跳過。")
            continue

        # 合併語音段
        speech_segments = [wav[ts['start']:ts['end']] for ts in speech_timestamps]
        combined = torch.cat(speech_segments)

        # 儲存到 output_root
        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file)
        torchaudio.save(output_path, combined.unsqueeze(0), sampling_rate)

        print(f"已儲存至：{output_path}")
