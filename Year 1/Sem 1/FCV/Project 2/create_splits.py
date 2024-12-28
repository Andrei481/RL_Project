import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
images_dir = "all"  # Directory containing all original images (JPEGs)
labels_with_face = "with_face"  # Directory with annotations for "with face"
labels_without_face = "without_face"  # Directory with annotations for "without face"
output_images_dir = "data/images"
output_labels_dir = "data/labels"

# Split Ratios
split_ratios = (0.8, 0.1, 0.1)  # Train, Val, Test

# Create Output Directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(output_labels_dir, split), exist_ok=True)

def get_image_label_pairs(labels_dir, images_dir, prefix):
    """
    Retrieve pairs of image and label file paths and attach a prefix to identify category.
    """
    labels = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    label_paths = [os.path.join(labels_dir, label) for label in labels]
    image_paths = [os.path.join(images_dir, os.path.splitext(label)[0] + ".jpg") for label in labels]
    return [(img, lbl, prefix) for img, lbl in zip(image_paths, label_paths)]

def split_data(data, split_ratios):
    """
    Split data into train, validation, and test sets.
    """
    train_data, temp_data = train_test_split(data, test_size=1 - split_ratios[0], random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]), random_state=42)
    return train_data, val_data, test_data

def copy_and_rename(data, split, output_images_dir, output_labels_dir):
    """
    Copy images and labels to their respective split directories, renaming files with a prefix.
    """
    for img_path, label_path, prefix in data:
        # Extract original names
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        
        # Create new names with prefix
        new_img_name = f"{prefix}_{img_name}"
        new_label_name = f"{prefix}_{label_name}"
        
        # Copy to split directories
        shutil.copy(img_path, os.path.join(output_images_dir, split, new_img_name))
        shutil.copy(label_path, os.path.join(output_labels_dir, split, new_label_name))

# Get image-label pairs for both categories
with_face_data = get_image_label_pairs(labels_with_face, images_dir, "with_face")
without_face_data = get_image_label_pairs(labels_without_face, images_dir, "without_face")

# Combine and shuffle the data for each category
train_with, val_with, test_with = split_data(with_face_data, split_ratios)
train_without, val_without, test_without = split_data(without_face_data, split_ratios)

# Combine the splits
train_data = train_with + train_without
val_data = val_with + val_without
test_data = test_with + test_without

# Shuffle the combined splits
import random
random.seed(42)
random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

# Copy and rename files
copy_and_rename(train_data, "train", output_images_dir, output_labels_dir)
copy_and_rename(val_data, "val", output_images_dir, output_labels_dir)
copy_and_rename(test_data, "test", output_images_dir, output_labels_dir)
