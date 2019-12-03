from PIL import Image, ImageDraw
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Activation, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import CSVLogger
import numpy as np
from random import randint
import os

IMAGE_WIDTH = 136
IMAGE_HEIGHT = 136
# MODEL_FILE = "../trainedModels/senesniModels/model5_nr_epochs_2.h5"
TEST_IM_NR = 10
NEW_RESULT_FOR_SAME_IMG = False
TEST_IM_PATH = '../testFaceDetectorImages/img' + str(TEST_IM_NR) +  '.jpg'
MIN_PROPOSE_REGION_SIZE = 10

def randomColour():
    return (randint(0,256), randint(0,256), randint(0,256))

def getModel1():
    global modelNr
    modelNr = 1
    global modelName
    modelName = "model1"
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(136,136,3)))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    # opt pakeiciau
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

def getModel2():
    global modelNr
    modelNr = 2
    global modelName
    modelName = "model2"
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(136,136,3)))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation("relu"))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    # opt pakeiciau
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

def getModel3():
    global modelNr
    modelNr = 3
    global modelName
    modelName = "model3"
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(136,136,3)))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

def getModel4():
    global modelNr
    modelNr = 4
    global modelName
    modelName = "model4"
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(136,136,3)))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    opt = keras.optimizers.rmsprop(lr=0.00008, decay=1e-6)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def getImageFromPath(path):
    #return image.load_img(path)
    return Image.open(path)

def getArrayFromImage(image):
    return np.array(image)

def getImageFromArray(array):
    return Image.fromarray(array)

def resizeImage(img):
    if img.size[1] > 700:
        factor = img.size[1] / 500
        new_width = int(img.size[0] / factor)
        print('new_width: ' + str(new_width))
        img = img.resize((new_width, 500))
        return img
    else:
        return img

model = getModel1()
MODEL_FILE = "../trainedModels/model" + str(modelNr) +  ".h5"
model.load_weights(MODEL_FILE)
img = getImageFromPath(TEST_IM_PATH)
img = resizeImage(img)
selectiveSearchResult = img
img_array = getArrayFromImage(img)

if not os.path.isdir('../faceDetectorResults/model' + str(modelNr)):
    os.mkdir('../faceDetectorResults/model' + str(modelNr))
RESULT_IM_PATH = '../faceDetectorResults/model' +str(modelNr) + '/img' + str(TEST_IM_NR) + 'result'  + '_model' + str(modelNr) +  '.jpg'
SELECTIVE_SEARCH_RESULT_PATH = '../faceDetectorResults/model' +str(modelNr) + '/img' + str(TEST_IM_NR) + 'selectiveSearch'  + '_model' + str(modelNr) +  '.jpg'
RESULT_IM_PATH_NEW = '../faceDetectorResults/model' +str(modelNr) + '/img' + str(TEST_IM_NR) + 'result' + '_model' + str(modelNr) +  'b.jpg'

img_lbl, regions = selectivesearch.selective_search(img_array, scale=500, sigma=0.9, min_size=10)
print("Number of regions: " + str(len(regions)))
img = getImageFromArray(img_array)
d = ImageDraw.Draw(img)
line_color = (255, 0, 0)
selectiveSearchResultDraw = ImageDraw.Draw(selectiveSearchResult)

for region in regions:
    print('img.size: ' + str(img.size))
    print('img_array.shape: ' + str(img_array.shape))
    x = region['rect'][0]
    y = region['rect'][1]
    w = region['rect'][2]
    h = region['rect'][3]
    leftTop = (x, y)
    leftBottom = (x, y+h)
    rightBottom = (x+w, y+h)
    rightTop = (x+w, y)
    left = x
    print('left: ' + str(left))
    right = x+w
    print('right: ' + str(right))
    top = y
    print('top: ' + str(top))
    bottom = y+h
    print('bottom: ' + str(bottom))
    selectiveSearchResultDraw.line([leftTop, leftBottom, rightBottom, rightTop, leftTop], fill=line_color, width=1)


    if(right - left >= MIN_PROPOSE_REGION_SIZE and bottom - top >= MIN_PROPOSE_REGION_SIZE):
        #print('Proposed region size is valid')
        #proposed_region_arr = img_array[left:right, top:bottom]
        proposed_region_arr = img_array[top:bottom, left:right]
        proposed_region_img = Image.fromarray(proposed_region_arr)
        proposed_region_img = proposed_region_img.resize((136, 136))
        # print('proposed_region_img.size: ' + str(proposed_region_img.size))
        proposed_region_arr = np.array(proposed_region_img)
        print('proposed_region_arr.shape: ' + str(proposed_region_arr.shape))
        proposed_region_arr = proposed_region_arr[np.newaxis, :]
        print('proposed_region_arr.shape after newaxis: ' + str(proposed_region_arr.shape))
        prediction = model.predict(proposed_region_arr)
        prediction = prediction[0][0]
        # 1 is not face
        # 0 is face
        print('prediction: ' + str(prediction))
        threshold = np.float32(0.3)

        if(prediction < threshold):
        #if True:
            print('Found a face region')
            print('Probablity that the region is a face: ' + str(np.float32(1)-prediction))
            d.line([leftTop, leftBottom, rightBottom, rightTop, leftTop], fill=line_color, width=1)
            if NEW_RESULT_FOR_SAME_IMG:
                img.save(RESULT_IM_PATH_NEW)
            else:
                img.save(RESULT_IM_PATH)

            print('')
print('img.size: ' + str(img.size))
selectiveSearchResult.save(SELECTIVE_SEARCH_RESULT_PATH)


# test_input = np.array(test_not_face1_im_path)
# test_input = test_input[np.newaxis, :]
# prediction = model.predict(test_input)
# prediction = prediction[0][0]
# #print('pred type: ' + str(type(preds[0][0])))
# threshold = np.float32(0.89)
# if(prediction >= threshold):
#     print('Image is not of a face')
# else:
#     print('Image is of a face')
# print('Probablity that the image is a face: ' + str(np.float32(1)-prediction))
