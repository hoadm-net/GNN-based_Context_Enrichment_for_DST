#!/usr/bin/env python3
"""
Script để download và extract MultiWOZ 2.4 dataset
"""

import os
import json
import urllib.request
import zipfile
from io import BytesIO
import argparse
from pathlib import Path

class MultiWOZDownloader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # URL cho MultiWOZ 2.4
        self.dataset_url = "https://raw.githubusercontent.com/smartyfh/MultiWOZ2.4/main/data/MULTIWOZ2.4.zip"
        
    def download_dataset(self):
        """Download và extract MultiWOZ 2.4"""
        print("Downloading MultiWOZ 2.4 dataset...")
        
        try:
            # Download zip file
            response = urllib.request.urlopen(self.dataset_url)
            zip_data = BytesIO(response.read())
            
            # Extract zip
            with zipfile.ZipFile(zip_data) as zip_ref:
                zip_ref.extractall(self.data_dir)
                
            print(f"Dataset downloaded and extracted to {self.data_dir}")
            
            # Di chuyển files từ MULTIWOZ2.4 folder lên level trên
            multiwoz_dir = self.data_dir / "MULTIWOZ2.4"
            if multiwoz_dir.exists():
                for file_path in multiwoz_dir.iterdir():
                    if file_path.is_file():
                        file_path.rename(self.data_dir / file_path.name)
                        
                # Xóa folder rỗng
                multiwoz_dir.rmdir()
                
            self._verify_files()
            return True
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def _verify_files(self):
        """Verify essential files exist"""
        required_files = [
            "data.json",
            "ontology.json", 
            "valListFile.json",
            "testListFile.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
                
        if missing_files:
            print(f"Warning: Missing files: {missing_files}")
        else:
            print("All required files verified successfully!")
            
    def get_dataset_info(self):
        """Print dataset statistics"""
        data_file = self.data_dir / "data.json"
        
        if not data_file.exists():
            print("Dataset not found. Please download first.")
            return
            
        with open(data_file, 'r') as f:
            data = json.load(f)
            
        print(f"\n=== MultiWOZ 2.4 Dataset Info ===")
        print(f"Total dialogues: {len(data)}")
        
        # Count turns
        total_turns = 0
        for dialogue in data.values():
            total_turns += len(dialogue['log'])
            
        print(f"Total turns: {total_turns}")
        print(f"Average turns per dialogue: {total_turns / len(data):.1f}")
        
        # Domain statistics  
        domains = set()
        for dialogue in data.values():
            for domain in dialogue['goal'].keys():
                if domain not in ['topic', 'message', 'messageLen', 'eod']:
                    domains.add(domain)
                    
        print(f"Domains: {sorted(list(domains))}")
        
        # Load train/val/test splits
        splits = {}
        for split_name, file_name in [
            ('test', 'testListFile.json'),
            ('val', 'valListFile.json')
        ]:
            split_file = self.data_dir / file_name
            if split_file.exists():
                with open(split_file, 'r') as f:
                    splits[split_name] = [line.strip() for line in f]
                    
        # Calculate train split (remaining dialogues)
        test_ids = set(splits.get('test', []))
        val_ids = set(splits.get('val', []))
        all_ids = set(data.keys())
        train_ids = all_ids - test_ids - val_ids
        
        print(f"\nData splits:")
        print(f"- Train: {len(train_ids)} dialogues")
        print(f"- Validation: {len(val_ids)} dialogues")  
        print(f"- Test: {len(test_ids)} dialogues")

def main():
    parser = argparse.ArgumentParser(description='Download MultiWOZ 2.4 dataset')
    parser.add_argument('--data_dir', default='data/raw', 
                       help='Directory to save dataset')
    parser.add_argument('--info', action='store_true',
                       help='Show dataset info after download')
    
    args = parser.parse_args()
    
    downloader = MultiWOZDownloader(args.data_dir)
    
    if downloader.download_dataset():
        print("Download completed successfully!")
        
        if args.info:
            downloader.get_dataset_info()
    else:
        print("Download failed!")

if __name__ == "__main__":
    main()