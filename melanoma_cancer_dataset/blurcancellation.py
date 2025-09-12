import cv2
import os
import shutil

# Function to check if image is blurry using variance of Laplacian
def is_blurry(image_path, threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True  # treat unreadable image as blurry
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold

# Function to filter dataset
def filter_blurry_images(input_dir, output_dir, threshold=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        # Maintain subfolder structure
        rel_path = os.path.relpath(root, input_dir)
        save_path = os.path.join(output_dir, rel_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                if not is_blurry(file_path, threshold):
                    shutil.copy(file_path, save_path)

# Paths for train and test datasets
train_input = "dataset/train"
test_input = "dataset/test"
train_output = "dataset_filtered/train"
test_output = "dataset_filtered/test"

# Filter both train and test
filter_blurry_images(train_input, train_output, threshold=100)
filter_blurry_images(test_input, test_output, threshold=100)

print("Filtering complete. Sharp images saved in dataset_filtered/")
