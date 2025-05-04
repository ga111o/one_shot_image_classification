from PIL import Image, ImageDraw
import random
import math
import numpy as np

def add_soft_reflection(image_path, num_points=5, max_radius=50):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    draw = ImageDraw.Draw(img, 'RGBA')
    
    for _ in range(num_points):
        x = random.randint(0, width)
        y = random.randint(0, height)
        
        angle = random.uniform(0, 360)
        aspect_ratio = random.uniform(0.3, 0.7)
        
        brightness = random.randint(150, 255)
        
        for r in range(max_radius, 0, -1):
            distance_ratio = r / max_radius
            opacity = int(brightness * math.exp(-5 * distance_ratio ** 2))
            
            a = r 
            b = int(r * aspect_ratio) 
            
            for i in range(0, 360, 5):
                rad = math.radians(i + angle)
                x1 = x + a * math.cos(rad)
                y1 = y + b * math.sin(rad)
                
                small_radius = 10
                draw.ellipse([x1-small_radius, y1-small_radius, 
                            x1+small_radius, y1+small_radius],
                            fill=(255, 255, 255, opacity))
    return img


input_image = "image_base/155347 .jpg"
result = add_soft_reflection(input_image, num_points=3, max_radius=100)
result.save("temp.jpg")
