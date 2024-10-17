import os
import cv2
import tkinter as tk
from tkinter import filedialog

def select_directory():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

class Augmentation:
    def __init__(self, name, param):
        self.name = name
        self.param = param

    def apply(self, image):
        if self.name == "dummy":
            return image

def read_config_file(config_file):
    augmentations = []
    with open(config_file, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) > 1:
                    for i in range(0, len(parts), 2):
                        operation = parts[i]
                        value = parts[i+1]
                        augmentations.append(Augmentation(operation, value))
                elif len(parts) == 1:
                    operation, value = parts[0], None
                    augmentations.append(Augmentation(operation, value))
    return augmentations

def save_augmented_images(output_dir, img_name, augmented_img, augmentation_name, count):
    base_name = os.path.splitext(img_name)[0]
    new_name = f"{base_name}_{augmentation_name}_{count}.jpg"
    save_path = os.path.join(output_dir, new_name)
    cv2.imwrite(save_path, augmented_img)

def process_images(input_dir, augmentations):
    output_dir = input_dir + "_aug"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 1
    for img_name in os.listdir(input_dir):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)

            for aug in augmentations:
                augmented_img = aug.apply(img)
                save_augmented_images(output_dir, img_name, augmented_img, aug.name, count)
                count += 1

if __name__ == "__main__":
    input_dir = select_directory()
    
    augmentations = read_config_file("config.txt")

    if input_dir:
        process_images(input_dir, augmentations)
    else:
        print("No directory selected.")
