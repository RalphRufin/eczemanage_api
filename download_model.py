"""
Download model during Docker build on Hugging Face Spaces
"""
import os
import requests
import gc

R2_BASE_URL = "https://r2-worker.eczemanage.workers.dev"
OUTPUT_DIR = "./derm_foundation/"

def download_model():
    print("=" * 70)
    print("DOWNLOADING DERM FOUNDATION MODEL (Hugging Face Spaces Build)")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = [
        "saved_model.pb",
        "variables/variables.index",
        "variables/variables.data-00000-of-00001"
    ]
    
    for file_path in files:
        print(f"\n📥 Downloading {file_path}...")
        url = f"{R2_BASE_URL}/{file_path}"
        local_path = os.path.join(OUTPUT_DIR, file_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            with requests.get(url, stream=True, timeout=1800) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                chunk_count = 0
                
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=2*1024*1024):  # 2MB chunks (HF has RAM)
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            downloaded += len(chunk)
                            chunk_count += 1
                            
                            if chunk_count % 5 == 0:
                                gc.collect()
                            
                            if total_size > 0 and chunk_count % 10 == 0:
                                progress = (downloaded / total_size) * 100
                                mb_downloaded = downloaded / (1024*1024)
                                mb_total = total_size / (1024*1024)
                                print(f"  Progress: {progress:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                
                gc.collect()
            
            print(f"✅ Successfully downloaded: {file_path}")
            
        except Exception as e:
            print(f"❌ Error downloading {file_path}: {e}")
            raise
    
    print("\n" + "=" * 70)
    print("✅ MODEL DOWNLOAD COMPLETE! Ready to serve predictions.")
    print("=" * 70)

if __name__ == "__main__":
    download_model()
