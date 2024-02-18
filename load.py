import os
import random

dataset_path = "dataset/"
train_split = 0.8
val_split = 0.1
test_split = 1 - train_split - val_split

os.makedirs(os.path.join(dataset_path, "train/images"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "val/images"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "test/images"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "train/labels"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "val/labels"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "test/labels"), exist_ok=True)

image_paths = []
annotation_paths = []

for sequence_type in ["daySequence", "nightSequence"]:
    sequence_path = os.path.join(dataset_path, sequence_type)
    class_folders = [f for f in os.listdir(sequence_path) if os.path.isdir(os.path.join(sequence_path, f))]

    for class_folder in class_folders:
        class_path = os.path.join(sequence_path, class_folder)
        image_folder = os.path.join(class_path, class_folder)
        annotation_folder = os.path.join(class_path, "Annotations", "Annotations")

        image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
        annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".txt")]

        for image_file, annotation_file in zip(image_files, annotation_files):
            image_path = os.path.join(image_folder, image_file)
            annotation_path = os.path.join(annotation_folder, annotation_file)
            image_paths.append(image_path)
            annotation_paths.append(annotation_path)

data_pairs = list(zip(image_paths, annotation_paths))

random.shuffle(data_pairs)

train_size = int(len(data_pairs)*train_split)
val_size = int(len(data_pairs)*val_split)
test_size = len(data_pairs) - train_size - val_size

train_pairs = data_pairs[:train_size]
val_pairs = data_pairs[train_size:train_size + val_size]
test_pairs = data_pairs[train_size+val_size:]

for image_path, annotation_path in train_pairs:
    image_filename = os.path.basename(image_path)
    annotation_filename = os.path.basename(annotation_path)
    os.rename(image_path, os.path.join(dataset_path, "train/images", image_filename))
    os.rename(annotation_path, os.path.join(dataset_path, "train/labels", annotation_filename))

for image_path, annotation_path in val_pairs:
    image_filename = os.path.basename(image_path)
    annotation_filename = os.path.basename(annotation_path)
    os.rename(image_path, os.path.join(dataset_path, "val/images", image_filename))
    os.rename(annotation_path, os.path.join(dataset_path, "val/labels", annotation_filename))

for image_path, annotation_path in test_pairs:
    image_filename = os.path.basename(image_path)
    annotation_filename = os.path.basename(annotation_path)
    os.rename(image_path, os.path.join(dataset_path, "test/images", image_filename))
    os.rename(annotation_path, os.path.join(dataset_path, "test/labels", annotation_filename))

print("Dataset organized successfully!")