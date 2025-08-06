# 建立虛擬環境及安裝套件
```
python3.12 -m venv whisper-env
source whisper-env/bin/activate
pip install -r requirements.txt
```
# 訓練語料分割
1. 將下載的`FSR-2025-Hakka-train`資料夾放置在`dataset`資料夾
2. 把音檔移動到`Corpus`資料夾，更換名稱為"taipu-train"/"zhaoan-train"
3. 執行以下指令進行資料集分割
```
python3 dataset/data_split.py --accent taipu --type pinyin
python3 dataset/data_split.py --accent taipu --type char
python3 dataset/data_split.py --accent zhaoan --type pinyin
python3 dataset/data_split.py --accent zhaoan --type char
```
# 訓練腳本
## 腳本說明
- `train/baseline.py`/ `train/finetune_with_lora.py`: whisper微調主程式
- `train/get_model.py`: 提取微調後特定的checkpoint，準備進行測試集推論
- `evaluate/inference.py`: 測試集推論程式

## 訓練指令
- 不使用LoRA進行訓練
```
./run_baseline.sh
```
- 使用LoRA進行訓練
```
./run_lora.sh
```
> 訓練拼音模型，設定 --language en; 訓練漢字模型，設定 --language zh

> 中斷訓練後要接續訓練，可以將腳本中`--resume_from_checkpoint`註解拿掉，會從最新的checkpoint接續訓練。

# 使用MUSAN進行語料混躁
- download muson
```
wget https://www.openslr.org/resources/17/musan.tar.gz
```
- 解壓縮
```
tar -xzvf musan.tar.gz
```
- 進行語音混躁
```
python3 utlis/add_noise.py
```