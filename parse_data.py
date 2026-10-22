import os
import numpy as np
import pandas as pd


FER2013_PATH = 'data/fer2013.csv'
FERPLUS_PATH = 'data/fer2013new.csv'
OUT_DIR = 'data'

def parse_ferplus():
    print("Đang đọc dữ liệu CSV...")
    df_pixels = pd.read_csv(FER2013_PATH)
    df_labels = pd.read_csv(FERPLUS_PATH)
    
    # Ghép bảng
    df = df_pixels.join(df_labels, rsuffix='_ferplus')
    
    vote_cols = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    splits = {
        'Training': {'images': [], 'labels': []},
        'PublicTest': {'images': [], 'labels': []},
        'PrivateTest': {'images': [], 'labels': []}
    }
    
    print("Đang lọc nhãn majority vote và xử lý ảnh...")
    for _, row in df.iterrows():
        # Loại mẫu Unknown hoặc NF
        if row.get('unknown', 0) > 0 or row.get('NF', 0) > 0:
            continue
            
        votes = row[vote_cols].values.astype(int)
        if np.max(votes) < 1:
            continue
            
        label = np.argmax(votes)
        
        # Chuyển string pixel thành mảng 48x48
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8).reshape(48, 48)
        # Nhân 3 kênh
        img_3c = np.stack([pixels] * 3, axis=-1)
        
        split_name = row['Usage']
        if split_name in splits:
            splits[split_name]['images'].append(img_3c)
            splits[split_name]['labels'].append(label)
            
    # Lưu file .npz
    mapping = {'Training': 'train', 'PublicTest': 'val', 'PrivateTest': 'test'}
    for k, v in splits.items():
        name = mapping[k]
        imgs = np.array(v['images'], dtype=np.uint8)
        labels = np.array(v['labels'], dtype=np.int64)
        
        out_path = os.path.join(OUT_DIR, f"{name}.npz")
        np.savez_compressed(out_path, images=imgs, labels=labels)
        print(f"Đã lưu {name}: {imgs.shape[0]} mẫu -> {out_path}")

if __name__ == '__main__':
    parse_ferplus()