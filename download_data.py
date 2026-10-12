"""Download the RouterBench 0-shot file (~100 MB) from Hugging Face into data/."""

import urllib.request

from router.data import DATA_DIR, DATASET

URL = "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/routerbench_0shot.pkl"

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    if DATASET.exists():
        print(f"Already downloaded: {DATASET}")
    else:
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, DATASET)
        print(f"Saved to {DATASET}")
