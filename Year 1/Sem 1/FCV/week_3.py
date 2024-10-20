import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import math

def select_directory():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

class Augmentation:
    def __init__(self, operations):
        self.operations = operations 

    def apply(self, image):
        for operation, param in self.operations:
            if operation == "dummy":
                continue
            elif operation == "gamma":
                image = self.apply_gamma_correction(image, param)
            elif operation == "brightness":
                image = self.apply_brightness(image, param)
            elif operation == "box_blur":
                image = self.apply_box_blur(image, param)
            elif operation == "contrast":
                image = self.apply_contrast(image, param)
            elif operation == "resize":
                image = self.apply_resize(image, param)
            elif operation == "rotation":
                image = self.apply_rotation(image, param)
            elif operation == "scaling":
                image = self.apply_scaling(image, param)
            elif operation == "shearing":
                image = self.apply_shearing(image, param)
            elif operation == "flipping":
                image = self.apply_flipping(image, param)
        return image

    def apply_gamma_correction(self, image, gamma):
        invGamma = 1.0 / float(gamma)
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_brightness(self, image, brightness):
        brightness = float(brightness)
        new_image = np.zeros(image.shape, image.dtype)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y, x, c] = np.clip(image[y, x, c] + brightness, 0, 255)
        return new_image

    def apply_box_blur(self, image, kernel_size):
        kernel_size = int(kernel_size)
        return cv2.blur(image, (kernel_size, kernel_size))

    def apply_contrast(self, image, contrast):
        contrast = float(contrast)
        return cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    def apply_resize(self, image, size):
        width, height = map(int, size.split())
        return cv2.resize(image, (width, height))
    
    def get_rotated_image_size(self, w, h, angle_rad):
        new_w = int(abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad)))
        new_h = int(abs(w * math.sin(angle_rad)) + abs(h * math.cos(angle_rad)))
        return new_w, new_h

    def apply_rotation(self, image, angle):
        angle = float(angle)
        (h, w) = image.shape[:2]
        angle_rad = math.radians(angle)

        new_w, new_h = self.get_rotated_image_size(w, h, angle_rad)

        rotated_img = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)

        new_center = (new_w // 2, new_h // 2)
        old_center = (w // 2, h // 2)

        for y in range(new_h):
            for x in range(new_w):
                x_shifted = x - new_center[0]
                y_shifted = y - new_center[1]

                old_x = int(x_shifted * math.cos(-angle_rad) - y_shifted * math.sin(-angle_rad)) + old_center[0]
                old_y = int(x_shifted * math.sin(-angle_rad) + y_shifted * math.cos(-angle_rad)) + old_center[1]

                if 0 <= old_x < w and 0 <= old_y < h:
                    rotated_img[y, x] = image[old_y, old_x]

        return rotated_img


    def apply_scaling(self, image, scale_factor):
        scale_factor = float(scale_factor)
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(image, (new_width, new_height))

    def apply_shearing(self, image, shear_factor):
        shear_factor = float(shear_factor)
        height, width = image.shape[:2]
        M = np.array([[1, shear_factor, 0],
                      [0, 1, 0]], dtype=float)
        return cv2.warpAffine(image, M, (width, height))

    def apply_flipping(self, image, flip_code):
        flip_code = int(flip_code)
        return cv2.flip(image, flip_code)

def read_config_file(config_file):
    augmentations = []
    with open(config_file, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                operations = []
                for i in range(0, len(parts), 2):
                    operation = parts[i]
                    if i+1 < len(parts):
                        param = parts[i+1]
                        operations.append((operation, param))
                    else:
                        operations.append((operation, None))
                augmentations.append(Augmentation(operations))
    return augmentations

def save_augmented_images(output_dir, img_name, augmented_img, augmentation_chain, count):
    base_name = os.path.splitext(img_name)[0]
    chain_name = "_".join([op for op, _ in augmentation_chain])
    new_name = f"{base_name}_{chain_name}_{count}.jpg"
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
                save_augmented_images(output_dir, img_name, augmented_img, aug.operations, count)
                count += 1

if __name__ == "__main__":
    input_dir = select_directory()
    
    augmentations = read_config_file("config.txt")

    if input_dir:
        process_images(input_dir, augmentations)
    else:
        print("No directory selected.")
