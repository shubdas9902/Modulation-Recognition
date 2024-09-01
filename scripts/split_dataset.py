import shutil
import os
import random

def split_dataset(source_dir, dest_dir, split_ratio=(0.8, 0.1, 0.1)):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        random.shuffle(files)
        num_files = len(files)
        train_end = int(num_files * split_ratio[0])
        val_end = int(num_files * (split_ratio[0] + split_ratio[1]))
        
        train_dir = os.path.join(dest_dir, 'train', cls)
        val_dir = os.path.join(dest_dir, 'validation', cls)
        test_dir = os.path.join(dest_dir, 'test', cls)
        
        for d in [train_dir, val_dir, test_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
        
        for i, file in enumerate(files):
            src = os.path.join(cls_dir, file)
            if i < train_end:
                dst = os.path.join(train_dir, file)
            elif i < val_end:
                dst = os.path.join(val_dir, file)
            else:
                dst = os.path.join(test_dir, file)
            shutil.copy(src, dst)

source_directory = 'spectrograms'
destination_directory = 'dataset_split'
split_dataset(source_directory, destination_directory)
