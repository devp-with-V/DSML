import os
import requests

# Dictionary mapping local filenames to reliable raw GitHub mirror URLs
# (Kaggle blocks automated requests without an API key by redirecting to a login page)
DATASETS = {
    "IBM_HR_Attrition.csv": "https://raw.githubusercontent.com/pavopax/ibm-hr-analytics-attrition-dataset/master/WA_Fn-UseC_-HR-Employee-Attrition.csv",
    "Mall_Customers.csv": "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv",
    "AB_NYC_2019.csv": "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AB_NYC_2019.csv"
}

def download_datasets(target_dir="data"):
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Downloading datasets to {os.path.abspath(target_dir)}...\n")
    
    for filename, url in DATASETS.items():
        filepath = os.path.join(target_dir, filename)
        if os.path.exists(filepath):
            print(f"[SKIP] {filename} already exists.")
            continue
            
        print(f"Downloading {filename}...")
        try:
            # stream=True handles larger files gracefully without spiking memory
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f" -> Success!")
        except Exception as e:
            print(f" -> Failed to download {filename}: {e}")
            
    print("\n--- Manual Downloads Required ---")
    print("1. Credit Card Fraud: Too large for raw GitHub (~150MB).")
    print("   Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("2. Online Retail: Often hosted as an Excel file rather than CSV.")
    print("   Link: https://www.kaggle.com/datasets/carrie1/ecommerce-data")

if __name__ == "__main__":
    download_datasets()
