import os
from sklearn.model_selection import train_test_split
from shutil import move
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import argparse
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


def dir_nohidden(directory):
    ''''
    list directories not hidden (to avoid .DS_store and such when listing dir)
    '''
    dir = []
    for f in os.listdir(directory):
         if not f.startswith('.'):
            dir.append(f)
    return dir



def split_folder_to_train_test_valid(data_directory):
    """
    Splits data from the original directory into train, test, and validation directories for each class.
    
    Args:
        data_directory (str): Path to the original data directory containing subdirectories for each class.
    """

    original_data_dir = '../data/cell_images'
    train_dir = '../data/cell_images/train'
    test_dir = '../data/cell_images/test'
    validation_dir = '../data/cell_images/validation'

    # Get a list of class subdirectories in the original data directory
    class_subdirectories = [d for d in os.listdir(original_data_dir) if os.path.isdir(os.path.join(original_data_dir, d))]

    for class_subdir in class_subdirectories: # for 'Class' in ['Class1', 'Class2']
        class_path = os.path.join(original_data_dir, class_subdir) # ../data/cell_images/Parasitized and ../data/cell_images/Uninfected
        class_images = [img for img in os.listdir(class_path)] # ['C13NThinF_IMG_20150614_131318_cell_179.png'... all images for each class

        train_images, test_validation_images = train_test_split(class_images, test_size=0.3, random_state=42)
        test_images, validation_images = train_test_split(test_validation_images, test_size=0.5, random_state=42)

        print(f'Total images for class {class_subdir}: ', len(class_images)) # 13780 (all images for class_subdir) or 100%
        print(f'Total train images: {len(train_images)} or {(len(train_images)/len(class_images))*100}% ') # 9646 (70% of all images from class_images)
        print(f'Total test images: {len(test_images)} or {len(test_images)/len(class_images)*100}% ') # 2067 class 1 15%
        print(f'Total validation images: {len(validation_images)} or {len(validation_images)/len(class_images)*100}% ')
        print('\n')


        # Create Class 1 and Class 2 subdirectories inside new folders
        train_class_dir = os.path.join(train_dir, class_subdir)
        os.makedirs(train_class_dir, exist_ok=True)

        test_class_dir = os.path.join(test_dir, class_subdir)
        os.makedirs(test_class_dir, exist_ok=True)

        validation_class_dir = os.path.join(validation_dir, class_subdir)
        os.makedirs(validation_class_dir, exist_ok=True)

        # Move images from each class folder to new subdirectories
        for img in train_images:
            src_path =os.path.join(class_path, img)
            dst_path =os.path.join(train_class_dir,img)
            shutil.copy(src_path, dst_path)


        for img in test_images:
            src_path =os.path.join(class_path, img)
            dst_path =os.path.join(test_class_dir,img)
            shutil.copy(src_path, dst_path)


        for img in validation_images:
            src_path =os.path.join(class_path, img)
            dst_path =os.path.join(validation_class_dir,img)
            shutil.copy(src_path, dst_path)

    folder_dir = []
    for x in ['train', 'test', 'validation']: folder_dir.append((os.path.join('../data/cell_images/', x)))

    for i in folder_dir:
        for name in os.listdir(i):
            print(i, name ,len(os.listdir(os.path.join(i,name))))
        # Print new sizes for each new folder
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Split data from original directory into train, test, and validation directories.")
        parser.add_argument("data_directory", type=str, help="Path to the original data directory")
        args = parser.parse_args()
        
        split_folder_to_train_test_valid(args.data_directory)