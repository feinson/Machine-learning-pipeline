import os
from PIL import Image
import numpy as np
from pathlib import Path

import time

def resize_image(img, fixed_height):
    height_propotion = fixed_height / img.size[1]
    new_width = int(img.size[0] * height_propotion)
    new_img = img.resize((new_width, fixed_height))
    return new_img

#first for loop is to determine the smallest height
if __name__ == "__main__":
    x = time.time()
    processed_images_path = "./data/all_processed_images"

    try:
        os.mkdir(processed_images_path)
    except FileExistsError:
        pass

    smallest_height = 10000
    pathlist = Path("./data/images").glob('**/*.png')
    list_of_images = []

    for path in pathlist:
        img = Image.open(path)
        img.filename=path.name
        list_of_images.append(img)
        if img.size[1] < smallest_height:
            smallest_height = img.size[1]


    for img in list_of_images:
        new_img = resize_image(img, smallest_height)
        if img.mode == "RGB":
            new_img.save(f"{processed_images_path}/{img.filename}")
        else:
            print(f"An image called {img.filename} was unfortuneatly not in RGB format, but rather in '{img.mode}' format. It has not been saved for this reason.")

    print("_________")
    print((time.time()-x))




