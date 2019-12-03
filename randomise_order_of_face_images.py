from PIL import Image
from keras.preprocessing import image
import os
import numpy as np
from random import randint

# np.random.seed(0)
ORIGINAL_IMAGES_PATH = 'cropped_face_images/'
NEW_DIR = 'cropped_face_images_random_order/'

if not os.path.isdir(NEW_DIR):
    os.mkdir(NEW_DIR)

imPaths = []
for dir in os.listdir(ORIGINAL_IMAGES_PATH):
    for f in os.listdir(ORIGINAL_IMAGES_PATH + dir):
        if(f.endswith(('.jpg', '.png'))):
            im_file_path = ORIGINAL_IMAGES_PATH + dir + '/' + f
            imPaths.append(im_file_path)

imNr = 1
while len(imPaths) > 0:
    randomIndex = randint(0, len(imPaths)-1)
    im_file_path = imPaths[randomIndex]
    im = image.load_img(im_file_path)
    fn, ext = os.path.splitext(f)
    im.save(NEW_DIR + str(imNr) + ext)
    imNr += 1
    imPaths.pop(randomIndex)
