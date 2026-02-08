import gdown
import os
import zipfile

def download_and_unzip(folder_url, download_path):
    # 1. Download the folder using gdown
    print(f"Starting download from: {folder_url}")
    # gdown will download the folder and its contents
    gdown.download_folder(folder_url, output=download_path, quiet=False, remaining_ok=True)
    
    print("\nDownload complete. Starting extraction...")

    # 2. Walk through the downloaded folder and unzip everything
    for root, dirs, files in os.walk(download_path):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)
                # Create a destination folder name (strip .zip)
                extract_to = os.path.join(root, file[:-4])
                
                print(f"Extracting: {file} -> {extract_to}")
                
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                    
                    # Optional: Remove the zip file after extraction to save space
                    # os.remove(file_path)
                except Exception as e:
                    print(f"Failed to unzip {file}: {e}")

if __name__ == "__main__":
    # REPLACE with your Google Drive folder URL
    GDRIVE_URL = "https://drive.google.com/drive/folders/1EmBKIpGt-5yvCHbuLvDWSHu_hvSK3_Dz"
    
    LOCAL_DIR = './Data/'
    
    download_and_unzip(GDRIVE_URL, LOCAL_DIR)
    print("\nAll done!")