from PIL import Image
from keras.preprocessing import image
import os
import numpy as np

FACES_EXTRACTED_PATH = 'cropped_face_images/'
ORIGINAL_IMAGES_PATH = 'lfw/'
os.mkdir(FACES_EXTRACTED_PATH)

for dir in os.listdir(ORIGINAL_IMAGES_PATH):
    os.mkdir(FACES_EXTRACTED_PATH + dir)
    for f in os.listdir(ORIGINAL_IMAGES_PATH + dir):
        if(f.endswith(('.jpg', '.png'))):
            im_file_path = ORIGINAL_IMAGES_PATH + dir + '/' + f
            im = image.load_img(im_file_path, target_size=(250,250))
            arrimg = np.array(im)
            cropped_arr = arrimg[57:193, 57:193]
            cropped_im = Image.fromarray(cropped_arr)
            cropped_im.save(FACES_EXTRACTED_PATH + dir + '/' + f)
