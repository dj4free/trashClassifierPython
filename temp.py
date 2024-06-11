import os


def rename_files(directory):
    """
    Rename all files in the specified directory by replacing spaces with underscores.

    Args:
        directory (str): Path to the directory containing files to rename.
    """
    for filename in os.listdir(directory):
        if ' ' in filename:
            new_filename = filename.replace(' ', '_')
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f'Renamed: {filename} -> {new_filename}')
        else:
            print(f'No change needed: {filename}')


if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    rename_files(directory)
