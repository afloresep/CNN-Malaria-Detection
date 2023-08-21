import os
from sklearn.model_selection import train_test_split
from shutil import move
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

import shutil
from sklearn.model_selection import train_test_split

def plot_sample_images(directory, num_samples=5):
            classes = sorted(os.listdir(directory)[1:])
            fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 10))


            for i, cls in enumerate(classes):
                class_dir = os.path.join(directory, cls)
                class_images = os.listdir(class_dir)[:num_samples]
                for j, image_name in enumerate(class_images):
                    image_path = os.path.join(class_dir, image_name)
                    img = load_img(image_path, target_size=(224, 224))
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    if j == 0:
                        axes[i, j].set_title(cls)
            plt.show()




def split_folder_to_train_test_valid(data_directory):
    # Define paths to the original data and the new train/test/validation directories
    original_data_dir = data_directory
    train_dir = original_data_dir+'/train'
    test_dir = original_data_dir+'/test'
    validation_dir = original_data_dir+'/validation'

    # Create the new train/test/validation directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Get a list of class subdirectories in the original data directory
    class_subdirectories = [d for d in os.listdir(original_data_dir) if os.path.isdir(os.path.join(original_data_dir, d))]

    # Split the data for each class into train, test, and validation sets
    for class_subdir in class_subdirectories:
        class_path = os.path.join(original_data_dir, class_subdir)
        class_images = [img for img in os.listdir(class_path) if img.endswith('.png')]
        
        # Split the images for the current class into train, test, and validation sets (70% 15% 15%)
        train_images, test_validation_images = train_test_split(class_images, test_size=0.3, random_state=42)
        test_images, validation_images = train_test_split(test_validation_images, test_size=0.5, random_state=42)
        
        # Create class subdirectories in train, test, and validation directories
        train_class_dir = os.path.join(train_dir, class_subdir)
        test_class_dir = os.path.join(test_dir, class_subdir)
        validation_class_dir = os.path.join(validation_dir, class_subdir)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        os.makedirs(validation_class_dir, exist_ok=True)
        
        # Move images to the appropriate directories
        for img in train_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(train_class_dir, img)
            shutil.copy(src_path, dst_path)
            
        for img in test_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(test_class_dir, img)
            shutil.copy(src_path, dst_path)
            
        for img in validation_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(validation_class_dir, img)
            shutil.copy(src_path, dst_path)
