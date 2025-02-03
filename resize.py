from PIL import Image
import os

def resize_images():
    input_dir = "image_user"
    output_dir = "image_user_resize"
    target_width = 640

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
    
            input_path = os.path.join(input_dir, filename)
            img = Image.open(input_path)
    
            original_width, original_height = img.size
    
            ratio = target_width / original_width
            new_height = int(original_height * ratio)
    
            resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
    
            output_path = os.path.join(output_dir, filename)
            resized_img.save(output_path, quality=95)

if __name__ == "__main__":
    resize_images()
