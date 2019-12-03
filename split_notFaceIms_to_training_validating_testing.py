from PIL import Image
from keras.preprocessing import image
import os
import numpy as np
from random import randint

# np.random.seed(0)
ALL_IMS_DIR = 'notFacesImsRandomOrder/'
TRAIN_DIR = 'train/'
VALIDATE_DIR = 'validate/'
TEST_DIR = 'test/'
NR_TRAIN = 10433
NR_VALIDATION = 2200
NR_TEST = 600

if not os.path.isdir(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)
if not os.path.isdir(VALIDATE_DIR):
    os.mkdir(VALIDATE_DIR)
if not os.path.isdir(TEST_DIR):
    os.mkdir(TEST_DIR)

counter = 0
for f in os.listdir(ALL_IMS_DIR):
    if(f.endswith(('.jpg', '.png'))):
        im_file_path = ALL_IMS_DIR  + f
        im = image.load_img(im_file_path)
        if counter < NR_TRAIN:
            im.save(TRAIN_DIR + f)
        elif counter < (NR_VALIDATION + NR_TRAIN):
            im.save(VALIDATE_DIR + f)
        else:
            im.save(TEST_DIR + f)
        counter += 1
