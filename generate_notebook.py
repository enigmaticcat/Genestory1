import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Đếm số lượng giá trị OL và X trong các tệp CSV của PROVEDIt\n",
                "Notebook này tính toán số lần xuất hiện của các giá trị `OL` và `X` trong tất cả các tệp CSV 1-5 person ở các mức số giây chạy khác nhau cho cả 5 thư mục thí nghiệm."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "\n",
                "base_dir = '/Users/nguyenthithutam/Desktop/TAWSEEM/PROVEDIt_1-5-Person CSVs Filtered'"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def count_values(main_folder):\n",
                "    main_path = os.path.join(base_dir, main_folder)\n",
                "    if not os.path.isdir(main_path):\n",
                "        print(f'{main_folder} not found.')\n",
                "        return\n",
                "    \n",
                "    results = {}\n",
                "    print(f'=== Kết quả cho thư mục: {main_folder} ===')\n",
                "    for root, dirs, files in os.walk(main_path):\n",
                "        for file in files:\n",
                "            if file.endswith('.csv'):\n",
                "                file_path = os.path.join(root, file)\n",
                "                try:\n",
                "                    df = pd.read_csv(file_path, low_memory=False)\n",
                "                    # Tính chính xác số lần xuất hiện 'OL' và 'X'\n",
                "                    ol_count = (df == 'OL').sum().sum()\n",
                "                    x_count = (df == 'X').sum().sum()\n",
                "                    \n",
                "                    rel_path = os.path.relpath(root, main_path)\n",
                "                    parts = rel_path.split(os.sep)\n",
                "                    if len(parts) >= 2:\n",
                "                        person_dir = parts[0]\n",
                "                        sec_dir = parts[1]\n",
                "                        if person_dir not in results:\n",
                "                            results[person_dir] = {}\n",
                "                        if sec_dir not in results[person_dir]:\n",
                "                            results[person_dir][sec_dir] = {'OL': 0, 'X': 0, 'files': 0}\n",
                "                        \n",
                "                        results[person_dir][sec_dir]['OL'] += int(ol_count)\n",
                "                        results[person_dir][sec_dir]['X'] += int(x_count)\n",
                "                        results[person_dir][sec_dir]['files'] += 1\n",
                "                except Exception as e:\n",
                "                    pass\n",
                "    \n",
                "    # In kết quả đã sắp xếp\n",
                "    for person in sorted(results.keys()):\n",
                "        print(f'  {person}:')\n",
                "        def get_sec(s):\n",
                "            try: return int(s.split()[0])\n",
                "            except: return 0\n",
                "        for sec in sorted(results[person].keys(), key=get_sec):\n",
                "            res = results[person][sec]\n",
                "            print(f'    {sec}: OL = {res[\"OL\"]}, X = {res[\"X\"]} (từ {res[\"files\"]} tệp CSV)')\n",
                "    print('\\n')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

main_folders = [
    "PROVEDIt_1-5-Person CSVs Filtered_3130_IDPlus28cycles",
    "PROVEDIt_1-5-Person CSVs Filtered_3130_PP16HS32cycles",
    "PROVEDIt_1-5-Person CSVs Filtered_3500_F6C29cycles_hlfrxn",
    "PROVEDIt_1-5-Person CSVs Filtered_3500_GF29cycles",
    "PROVEDIt_1-5-Person CSVs Filtered_3500_IDPlus29cycles"
]

for folder in main_folders:
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"count_values('{folder}')"]
    })

with open('count_OL_X.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
