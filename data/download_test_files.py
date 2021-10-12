import sys
from os.path import abspath, basename, dirname, exists

import wget

SCRIPT_DIR = dirname(abspath(__file__))

FILE_URLS = [
    (
        "B2K1Gamma_RapidSim_7TeV_K1KstarNonResonant_Tree.root",
        "https://cernbox.cern.ch/index.php/s/8mN10X8U7VGfaRc/download",
    ),
    (
        "B2K1Gamma_RapidSim_7TeV_Tree.root",
        "https://cernbox.cern.ch/index.php/s/pr3aM8n2hPT4Pag/download",
    ),
    (
        "B2KstGamma_RapidSim_7TeV_KstarNonResonant_Tree.root",
        "https://cernbox.cern.ch/index.php/s/QuP2cHeISTTSLVv/download",
    ),
    (
        "B2KstGamma_RapidSim_7TeV_Tree.root",
        "https://cernbox.cern.ch/index.php/s/EH5yrCpGko7P7Mc/download",
    ),
]


def progress_bar(current, total, width=80):
    percentage = current / total * 100
    current = int(current / 1e6)
    total = int(total / 1e6)
    progress_message = f"{current:,}MB / {total:,}MB ({percentage:2.1f}%) "
    sys.stdout.write("\r  " + progress_message)
    sys.stdout.flush()


def download(output_file, url):
    if exists(output_file):
        print(f"Already downloaded: {basename(output_file)}")
    else:
        print(f"Downloading: {basename(output_file)}")
        wget.download(url=url, bar=progress_bar, out=output_file)
        print()


if __name__ == "__main__":
    for output_file, url in FILE_URLS:
        output_file = f"{SCRIPT_DIR}/{output_file}"
        download(output_file, url)
