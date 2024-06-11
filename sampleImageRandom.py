import os
import random
import shutil


def clear_directory(directory):
    """
    Clears all files in the specified directory.

    Args:
        directory (str): Path to the directory to clear.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def copy_random_images(source_dirs, destination_dir, num_images=55):
    """
    Copies a specified number of random images from each source directory to a destination directory.

    Args:
        source_dirs (list): List of paths to source directories.
        destination_dir (str): Path to the destination dir.
        num_images (int): Number of images to copy from each source dir. Defaults to 55.
    """
    clear_directory(destination_dir)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
            random_images = random.sample(images, min(num_images, len(images)))

            for image in random_images:
                source_path = os.path.join(source_dir, image)
                dest_path = os.path.join(destination_dir, image)
                shutil.copy2(source_path, dest_path)
                print(f"Copied: {source_path} to {dest_path}")
        else:
            print(f"Source directory does not exist: {source_dir}")


if __name__ == "__main__":
    base_dir = 'data/realwaste/realwaste-main/RealWaste'
    source_dirs = [
        os.path.join(base_dir, 'Cardboard'),
        os.path.join(base_dir, 'Food_Organics'),
        os.path.join(base_dir, 'Glass'),
        os.path.join(base_dir, 'Metal'),
        os.path.join(base_dir, 'Miscellaneous_Trash'),
        os.path.join(base_dir, 'Paper'),
        os.path.join(base_dir, 'Plastic'),
        os.path.join(base_dir, 'Vegetation')
    ]
    destination_dir = 'SampleImages'

    copy_random_images(source_dirs, destination_dir)
