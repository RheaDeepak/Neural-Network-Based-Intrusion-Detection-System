import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

urls = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
    "KDDTest-21.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest-21.txt"
}

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    for filename, url in urls.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Successfully downloaded {filename}.")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print("Please download it manually and place it in the 'data/' folder.")
        else:
            print(f"{filename} already exists. Skipping download.")

if __name__ == "__main__":
    download_data()
