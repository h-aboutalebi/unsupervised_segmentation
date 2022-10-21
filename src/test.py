from PIL import Image
import os
from utils import get_transform
directory = "/home/hossein/data/mahdi_dataset/train_truck/directory/truck/imgs/val"

for i, filename in enumerate(os.listdir(directory)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        try:
            img = Image.open(os.path.join(directory, filename))
            transform = get_transform(224, False, crop_type = "center")
            transform(img)
        except Exception as e: 
            print(e)
            print("error in {}".format( os.path.join(directory, filename)))
