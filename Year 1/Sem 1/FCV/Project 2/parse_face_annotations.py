import os
import shutil

def process_text_files(source_dir, without_face_dir, with_face_dir):
    # Ensure the target directories exist
    os.makedirs(without_face_dir, exist_ok=True)
    os.makedirs(with_face_dir, exist_ok=True)

    # Process each file in the source directory
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        # Skip if not a file or not a text file
        if not os.path.isfile(file_path) or not filename.endswith('.txt'):
            continue

        # Check if the file is empty
        target_dir = without_face_dir if os.path.getsize(file_path) == 0 else with_face_dir
        # prefix = "without_face_" if os.path.getsize(file_path) == 0 else "with_face_"

        # Construct the new filename and destination path
        # new_filename = prefix + filename
        destination_path = os.path.join(target_dir, filename)

        # Move the file to the appropriate directory
        shutil.copy(file_path, destination_path)
        # print(f"Moved '{filename}' to '{target_dir}' as '{new_filename}'.")

if __name__ == "__main__":
    # Define source and target directories
    source_directory = "data/labels/train"  # Replace with your source directory
    without_face_directory = "data/labels/without_face"
    with_face_directory = "data/labels/with_face"

    process_text_files(source_directory, without_face_directory, with_face_directory)
