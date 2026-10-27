# download_data.py
import os
import wfdb
from config import DS1_TRAIN, DS2_TEST

def download_records(target_dir='mitdb'):
    os.makedirs(target_dir, exist_ok=True)
    all_records = DS1_TRAIN + DS2_TEST
    print(f"Checking {len(all_records)} records...")
    
    for rec in all_records:
        rec_path = os.path.join(target_dir, rec)
        if not (os.path.exists(f"{rec_path}.dat") and os.path.exists(f"{rec_path}.atr")):
            print(f"Downloading record {rec}...")
            wfdb.dl_files('mitdb', target_dir, [f'{rec}.dat', f'{rec}.hea', f'{rec}.atr'])
            
    print("Download complete.")

if __name__ == '__main__':
    download_records()