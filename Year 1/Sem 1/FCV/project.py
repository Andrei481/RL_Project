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

    def apply_rotation(self, image, angle):
        angle = float(angle)
        
        height, width = image.shape[:2]
        
        angle_rad = abs(math.radians(angle))
        new_width = int(width * abs(math.cos(angle_rad)) + height * abs(math.sin(angle_rad)))
        new_height = int(width * abs(math.sin(angle_rad)) + height * abs(math.cos(angle_rad)))
        
        center = (width/2, height/2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        
        return cv2.warpAffine(image, rotation_matrix, (new_width, new_height))


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
        (h, w) = image.shape[:2]
        flipped_image = np.zeros_like(image)

        if flip_code == 1:
            for y in range(h):
                for x in range(w):
                    flipped_image[y, x] = image[y, w - 1 - x]

        elif flip_code == 0:
            for y in range(h):
                flipped_image[y] = image[h - 1 - y]

        elif flip_code == -1:
            for y in range(h):
                for x in range(w):
                    flipped_image[y, x] = image[h - 1 - y, w - 1 - x]

        return flipped_image


def read_config_file(config_file):
    """
    Read and validate configuration file.
    Raises ValueError for invalid configurations.
    """
    augmentations = []
    
    # Check if file exists
    if not os.path.exists(config_file):
        raise ValueError(f"Configuration file not found: {config_file}")
        
    try:
        with open(config_file, "r") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) % 2 != 0:
                    raise ValueError(f"Line {line_num}: Each operation must have a parameter")
                
                operations = []
                for i in range(0, len(parts), 2):
                    operation = parts[i].lower()
                    param = parts[i+1]
                    
                    valid_operations = {"dummy", "gamma", "brightness", "box_blur", 
                                     "contrast", "rotation", "scaling", "shearing", "flipping"}
                    if operation not in valid_operations:
                        raise ValueError(f"Line {line_num}: Invalid operation '{operation}'")
                    
                    try:
                        if operation == "gamma":
                            gamma = float(param)
                            if gamma <= 0:
                                raise ValueError(f"Line {line_num}: Gamma must be positive")
                        elif operation == "brightness":
                            float(param) 
                        elif operation == "box_blur":
                            kernel = int(param)
                            if kernel <= 0 or kernel % 2 == 0:
                                raise ValueError(f"Line {line_num}: Kernel size must be positive odd number")
                        elif operation == "contrast":
                            contrast = float(param)
                            if contrast < 0:
                                raise ValueError(f"Line {line_num}: Contrast must be non-negative")
                        elif operation == "rotation":
                            float(param)
                        elif operation == "scaling":
                            scale = float(param)
                            if scale <= 0:
                                raise ValueError(f"Line {line_num}: Scale factor must be positive")
                        elif operation == "shearing":
                            float(param)
                        elif operation == "flipping":
                            flip = int(param)
                            if flip not in [-1, 0, 1]:
                                raise ValueError(f"Line {line_num}: Flip code must be -1, 0, or 1")
                        
                        operations.append((operation, param))
                    except ValueError as e:
                        if str(e).startswith("Line"):
                            raise
                        raise ValueError(f"Line {line_num}: Invalid parameter '{param}' for {operation}")
                
                augmentations.append(Augmentation(operations))
        
        if not augmentations:
            raise ValueError("Configuration file is empty or contains no valid operations")
            
        return augmentations
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error reading configuration file: {str(e)}")

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
    
    try:
        augmentations = read_config_file("config_3.txt")
        process_images(input_dir, augmentations)
    except ValueError as e:
        print(f"Configuration error: {e}")
        exit(1)
