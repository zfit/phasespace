import wget
from multiprocessing.pool import ThreadPool

files_urls = [
    ('B2K1Gamma_RapidSim_7TeV_K1KstarNonResonant_Tree.root',
     'https://cernbox.cern.ch/index.php/s/8mN10X8U7VGfaRc/download'),
    ('B2K1Gamma_RapidSim_7TeV_Tree.root', 'https://cernbox.cern.ch/index.php/s/pr3aM8n2hPT4Pag/download'),
    ('B2KstGamma_RapidSim_7TeV_KstarNonResonant_Tree.root',
     'https://cernbox.cern.ch/index.php/s/QuP2cHeISTTSLVv/download'),
    ('B2KstGamma_RapidSim_7TeV_Tree.root', 'https://cernbox.cern.ch/index.php/s/EH5yrCpGko7P7Mc/download')]


def download(url_file_name):
    file_name, url = url_file_name
    print(f"starting download of {file_name}")
    wget.download(url=url, bar=False, out=file_name)
    print("download finished")
    return file_name


if __name__ == '__main__':
    # files = ThreadPool(len(files_urls)).imap_unordered(download, files_urls)
    files = [download(url_file_name=url_file_name) for url_file_name in files_urls]
    for file in files:
        print(file)
