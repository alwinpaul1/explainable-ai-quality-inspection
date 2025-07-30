#!/usr/bin/env python3
"""
Dataset download script for quality inspection datasets.
Supports multiple popular manufacturing defect detection datasets.
"""

import os
import zipfile
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def download_file(url, filepath):
    """Download a file from URL with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=os.path.basename(filepath),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            pbar.update(size)

def download_casting_dataset(data_dir):
    """Download casting product quality inspection dataset from Kaggle."""
    print("Downloading casting product quality inspection dataset...")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download dataset
    dataset_name = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
    api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
    
    print(f"Dataset downloaded and extracted to: {data_dir}")

def download_mvtec_dataset(data_dir):
    """Download MVTec Anomaly Detection dataset."""
    print("Downloading MVTec AD dataset...")
    
    mvtec_dir = os.path.join(data_dir, "mvtec_ad")
    os.makedirs(mvtec_dir, exist_ok=True)
    
    # MVTec AD dataset URL
    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
    
    filename = os.path.join(mvtec_dir, "mvtec_anomaly_detection.tar.xz")
    
    print("Note: MVTec dataset is large (~4.9GB). This may take a while...")
    print("If download fails, please visit: https://www.mvtec.com/company/research/datasets/mvtec-ad")
    
    try:
        download_file(url, filename)
        print(f"Dataset downloaded to: {filename}")
        print("Please extract the .tar.xz file manually.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from https://www.mvtec.com/company/research/datasets/mvtec-ad")

def download_neu_dataset(data_dir):
    """Download NEU Surface Defect dataset."""
    print("Downloading NEU Surface Defect dataset...")
    
    neu_dir = os.path.join(data_dir, "neu_det")
    os.makedirs(neu_dir, exist_ok=True)
    
    print("NEU dataset requires manual download from:")
    print("http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html")
    print("Please download and place files in:", neu_dir)

def setup_dataset_structure(data_dir, dataset_type):
    """Setup proper directory structure for the dataset."""
    if dataset_type == "casting":
        # Expected structure after Kaggle download
        raw_dir = os.path.join(data_dir, "raw")
        processed_dir = os.path.join(data_dir, "processed")
        splits_dir = os.path.join(data_dir, "splits")
        
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(splits_dir, exist_ok=True)
        
        print(f"Dataset structure created in: {data_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download quality inspection datasets")
    parser.add_argument("--dataset", "-d", 
                       choices=["casting", "mvtec", "neu", "all"],
                       default="casting",
                       help="Dataset to download")
    parser.add_argument("--data-dir", 
                       default="../data",
                       help="Directory to store datasets")
    
    args = parser.parse_args()
    
    # Get absolute path
    data_dir = os.path.abspath(args.data_dir)
    
    print(f"Data directory: {data_dir}")
    
    if args.dataset == "casting" or args.dataset == "all":
        try:
            download_casting_dataset(data_dir)
            setup_dataset_structure(data_dir, "casting")
        except Exception as e:
            print(f"Error downloading casting dataset: {e}")
            print("Make sure you have configured Kaggle API credentials.")
            print("Run: kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product")
    
    if args.dataset == "mvtec" or args.dataset == "all":
        download_mvtec_dataset(data_dir)
    
    if args.dataset == "neu" or args.dataset == "all":
        download_neu_dataset(data_dir)
    
    print("\nDataset download complete!")
    print(f"Data stored in: {data_dir}")
    print("\nNext steps:")
    print("1. Activate virtual environment: source quality_env/bin/activate")
    print("2. Install requirements: pip install -r requirements.txt")
    print("3. Run training: python src/training/train_model.py")

if __name__ == "__main__":
    main()