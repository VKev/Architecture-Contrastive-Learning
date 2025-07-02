import os
import shutil
from scipy.io import loadmat

VAL_DIR = r'D:\AI\Dataset\ImageNet1k\val'
GT_FILE = r'D:\AI\Dataset\ImageNet1k\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt'
META_MAT = r'D:\AI\Dataset\ImageNet1k\ILSVRC2012_devkit_t12\data\meta.mat'

def load_wnid_mapping(meta_mat_path):
    """Load class index to wnid mapping from meta.mat."""
    meta = loadmat(meta_mat_path, squeeze_me=True)['synsets']
    mapping = {}
    for entry in meta:
        if entry['ILSVRC2012_ID'] <= 1000:
            mapping[entry['ILSVRC2012_ID']] = entry['WNID']
    return mapping

def prepare_validation_set(val_dir, ground_truth_file, class_mapping):
    with open(ground_truth_file, 'r') as f:
        labels = [int(x.strip()) for x in f.readlines()]

    img_files = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])
    if len(img_files) != len(labels):
        raise ValueError("Mismatch between image count and label count.")

    for img_name, label in zip(img_files, labels):
        wnid = class_mapping[label]
        class_dir = os.path.join(val_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)

        src_path = os.path.join(val_dir, img_name)
        dst_path = os.path.join(class_dir, img_name)
        shutil.move(src_path, dst_path)
        print(f"Moved {img_name} → {wnid}/")

if __name__ == '__main__':
    class_mapping = load_wnid_mapping(META_MAT)
    prepare_validation_set(VAL_DIR, GT_FILE, class_mapping)
