import argparse
import csv
import os
import re

# Define the main function
def main():
    parser = argparse.ArgumentParser(description="Process CSV files based on accent and type.")
    parser.add_argument("--accent", required=True, choices=["taipu", "zhaoan"], help="Accent type (e.g., taipu or zhaoan)")
    parser.add_argument("--type", required=True, choices=["pinyin", "char"], help="Data type (e.g., pinyin or char)")
    args = parser.parse_args()

    accent = args.accent
    data_type = args.type

    dic = {
        'taipu_eval': ['DF101J2003', 'DF103K2001', 'DF111K2001', 'DM101J2004', 'DM102K2002', 'DM115K2001'],
        'taipu_test': ['DF103L2005', 'DF139L2084', 'DF204L2059', 'DM102L2025', 'DM201K2001', 'DM202L2063'],
        'zhaoan_eval': ['ZF101Q2001', 'ZF102S2001', 'ZF113Q2003', 'ZM113P2101', 'ZM113R2101', 'ZM115R2038'],
        'zhaoan_test': ['ZF103P2005', 'ZF105R2009', 'ZF105S2009', 'ZM113R2016', 'ZM115Q2003', 'ZM119S2051']
    }

    csv_filename = {
        'taipu': ['訓練_DF_大埔腔_女_edit.csv', '訓練_DM_大埔腔_男_edit.csv'],
        'zhaoan': ['訓練_ZF_詔安腔_女_edit.csv', '訓練_ZM_詔安腔_男_edit.csv']
    }

    type_dic = {
        'pinyin': '客語拼音',
        'char': '客語漢字'
    }

    datasets = {"train": ([], []), "test": ([], []), "eval": ([], [])}

    for file in csv_filename[accent]:
        file = f"dataset/FSR-2025-Hakka-train/{file}"
        with open(file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.strip().replace('\ufeff', '') for name in reader.fieldnames]
            for row in reader:
                filename = row.get('檔名')
                hakka_pinyin = row.get(type_dic[data_type])

                if filename != '' and hakka_pinyin != '':
                    hakka_pinyin = re.sub(r'\s+', ' ', hakka_pinyin).strip()
                    for dataset_type in ["test", "eval"]:
                        if filename[:10] in dic.get(f'{accent}_{dataset_type}', []):
                            datasets[dataset_type][0].append(f"{filename[:14]} {f'Corpus/{accent}-train/{filename[:10]}/{filename}'}")
                            datasets[dataset_type][1].append(f"{filename[:14]} {hakka_pinyin}")
                            break
                    else:
                        datasets["train"][0].append(f"{filename[:14]} {f'Corpus/{accent}-train/{filename[:10]}/{filename}'}")
                        datasets["train"][1].append(f"{filename[:14]} {hakka_pinyin}")

    # Write to output files
    for dataset_type, (paths, texts) in datasets.items():
        datadir = f"dataset/train-{accent}-{data_type}/{dataset_type}"
        os.makedirs(datadir, exist_ok=True)
        with open(f"{datadir}/audio_paths", 'w', encoding='utf-8') as f:
            for line in paths:
                f.write(line + '\n')
        with open(f"{datadir}/text", 'w', encoding='utf-8') as f:
            for line in texts:
                f.write(line + '\n')


if __name__ == "__main__":
    main()