import os
import shutil
from pathlib import Path

def copy_images_to_destination():
    source_dir = "wikiart"
    dest_dir = "./images"
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                source_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                
                try:
                    shutil.copy2(source_file, dest_file)
                    print(f"Copied: {file}")
                except Exception as e:
                    print(f"Error copying {file}: {str(e)}")

if __name__ == "__main__":
    copy_images_to_destination()
