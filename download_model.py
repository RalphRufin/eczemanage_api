"""
Download model during Docker build
"""
import os
import requests

R2_BASE_URL = "https://r2-worker.eczemanage.workers.dev"
OUTPUT_DIR = "./derm_foundation/"

def download_model():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = [
        "saved_model.pb",
        "variables/variables.index",
        "variables/variables.data-00000-of-00001"
    ]
    
    for file_path in files:
        print(f"Downloading {file_path}...")
        url = f"{R2_BASE_URL}/{file_path}"
        local_path = os.path.join(OUTPUT_DIR, file_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with requests.get(url, stream=True, timeout=900) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):  # 1MB chunks for build
                    if chunk:
                        f.write(chunk)
        print(f"✓ Downloaded: {file_path}")

if __name__ == "__main__":
    download_model()