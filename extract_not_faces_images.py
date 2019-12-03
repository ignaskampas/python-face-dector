from PIL import Image
from keras.preprocessing import image
import os
import numpy as np
import random

# np.random.seed(0)
ORIGINAL_IMAGES_PATH = 'not_faces_deletedFaces/'
CHANGED_IMAGES = 'notFacesImsRandomRegions/'
MAX_NR_IMAGES = 13233

if not os.path.isdir(CHANGED_IMAGES):
    os.mkdir(CHANGED_IMAGES)

counter = 0

while counter < MAX_NR_IMAGES:
    for dir in os.listdir(ORIGINAL_IMAGES_PATH):
        if not os.path.isdir(CHANGED_IMAGES + dir):
            os.mkdir(CHANGED_IMAGES + dir)
        for f in os.listdir(ORIGINAL_IMAGES_PATH + dir):
            if(f.endswith(('.jpg', '.png'))):
                im_file_path = ORIGINAL_IMAGES_PATH + dir + '/' + f
                im = image.load_img(im_file_path)
                width, height = im.size
                #print('im width: ' + str(width))
                #print('im height: ' + str(height))
                arrimg = np.array(im)

                if(width < height):
                    minWH = width
                else:
                    minWH = height

                #print('minWH: ' + str(minWH))
                randomNr = random.uniform(0,1)
                if(randomNr < float(0.1)):
                    croppedSize = int((float(minWH)- float(minWH)*0.7) * random.uniform(0,1) + float(minWH)*0.06)
                else:
                    # float(minWH)*0.04 makes sure that croppedSize is always at least 0.04 of minWH
                    # float(minWH)*0.04) makes sure that croppedSize is not bigger than minWH
                    croppedSize = int((float(minWH)- float(minWH)*0.06) * random.uniform(0,1) + float(minWH)*0.06)

                #print('croppedSize: ' + str(croppedSize))
                lengthThatIsLeft = minWH - croppedSize
                #print('lengthThatIsLeft: ' + str(lengthThatIsLeft))
                widthStart = int(float(lengthThatIsLeft)*random.uniform(0,1))
                heightStart = int(float(lengthThatIsLeft)*random.uniform(0,1))
                #print('widthStart: ' + str(widthStart))
                #print('heightStart: ' + str(heightStart))
                widthEnd = widthStart + croppedSize
                heightEnd = heightStart + croppedSize
                #print('widthEnd: ' + str(widthEnd))
                #print('heightEnd: ' + str(heightEnd))

                cropped_arr = arrimg[widthStart:widthEnd, heightStart:heightEnd]
                cropped_im = Image.fromarray(cropped_arr)
                cropped_im = cropped_im.resize((136,136))
                if not os.path.exists(CHANGED_IMAGES + dir + '/' + f):
                    cropped_im.save(CHANGED_IMAGES + dir + '/' + f)
                else:
                    fn, ext = os.path.splitext(f)
                    cropped_im.save(CHANGED_IMAGES + dir + '/' + fn + 'a' + ext)
                counter += 1
                #print('counter: ' + str(counter))
                #print('')
                if(counter >= MAX_NR_IMAGES):
                    break
            if(counter >= MAX_NR_IMAGES):
                break
        if(counter >= MAX_NR_IMAGES):
            break
