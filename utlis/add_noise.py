import os
import random
import argparse
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

# 噪聲類別與 SNR 選項
noise_labels = ["noise", "speech", "music"]
snr_choices = [10, 15, 20]

# 讀取音檔並加上隨機噪音
def mix_audio(src_path, noise_files, out_root, noise_label, src_root):
    audio = AudioSegment.from_wav(src_path)
    combined_noise = AudioSegment.silent(duration=len(audio))
    selected = random.sample(noise_files, k=min(random.randint(2, 3), len(noise_files)))
    snr_db = random.choice(snr_choices)

    for noise_path in selected:
        noise = AudioSegment.from_wav(noise_path)
        if len(noise) < len(audio):
            noise = noise * (len(audio) // len(noise) + 1)
        noise = noise[:len(audio)]
        attenuate = audio.dBFS - noise.dBFS - snr_db
        noise = noise + attenuate
        combined_noise = combined_noise.overlay(noise)

    mixed = audio.overlay(combined_noise)

    rel = os.path.relpath(src_path, start=src_root)
    new_base = os.path.basename(src_path)
    out_path = os.path.join(out_root, os.path.dirname(rel), new_base)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mixed.export(out_path, format='wav')
    print(f"{rel} 混音完成（{noise_label}, SNR={snr_db}dB）→ {new_base}")

# 主函式
def main(input_dir):
    input_dir = os.path.abspath(input_dir)
    print(f"▶ 處理資料夾：{input_dir}")
    src_files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(input_dir)
        for f in fs if f.endswith(".wav")
    ]

    if not src_files:
        print("找不到任何 .wav 檔案！")
        return

    for noise_label in noise_labels:
        noise_root = f"musan/{noise_label}"
        out_root = f"{input_dir}_{noise_label}"
        os.makedirs(out_root, exist_ok=True)

        noise_files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(noise_root)
            for f in fs if f.endswith(".wav")
        ]
        if not noise_files:
            print(f"噪聲資料夾為空：{noise_root}")
            continue

        print(f"加入噪音類別：{noise_label}（{len(noise_files)} 個檔案） → {out_root}")

        max_workers = max(4, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(mix_audio, src, noise_files, out_root, noise_label, input_dir)
                for src in src_files
            ]
            for f in as_completed(futures):
                _ = f.result()

    print("噪音混音完成！")

# 執行入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將 MUSAN 噪聲加入語音資料夾")
    parser.add_argument("input_dir", type=str, help="輸入的語音資料夾（包含 .wav）")
    args = parser.parse_args()
    main(args.input_dir)
