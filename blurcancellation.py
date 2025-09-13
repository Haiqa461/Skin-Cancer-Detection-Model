import cv2
import os

# Function to check if image is blurry
def is_blurry(image_path, threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:  # unreadable file
        return True
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold

# Function to clean dataset in place (delete blurry images)
def clean_dataset(root_dir, threshold=100):
    deleted_count = 0
    kept_count = 0
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(subdir, file)
                if is_blurry(file_path, threshold):
                    os.remove(file_path)  # delete blurry image
                    deleted_count += 1
                else:
                    kept_count += 1
    print(f"Cleaning done in {root_dir}.")
    print(f"✔ Kept images: {kept_count}")
    print(f"✘ Deleted blurry images: {deleted_count}")

# Paths for train and test datasets
dataset_path = r"C:\Users\haiqa\Documents\AI_skin_cancer_detection_model\Skin-Cancer-Detection-Model\melanoma_cancer_dataset"
train_path = os.path.join(dataset_path, r"C:\Users\haiqa\Documents\AI_skin_cancer_detection_model\Skin-Cancer-Detection-Model\melanoma_cancer_dataset\train")
test_path = os.path.join(dataset_path, r"C:\Users\haiqa\Documents\AI_skin_cancer_detection_model\Skin-Cancer-Detection-Model\melanoma_cancer_dataset\test")

# Run cleaning on both train and test
clean_dataset(train_path, threshold=100)
clean_dataset(test_path, threshold=100)