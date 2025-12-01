import kaggle
import os

def download_paysim_data():
    kaggle.api.dataset_download_files('ealaxi/paysim1', path='./data/raw/', unzip=True)

if __name__ == "__main__":
    os.makedirs("./data/raw", exist_ok=True)
    download_paysim_data()
    print("Downloaded PaySim to data/raw/")
