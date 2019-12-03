import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import CSVLogger

trainingLogsPath = "../modelResults/trainingLogs.csv"
modelPathPrefix = "../trainedModels/model"
modelPathSuffix = ".h5"

imgWidth = 136
imgHeight = 136
batch_size = 32
modelName = ""

trainingDataDir = "../train"
validationDataDir = "../validate"

# 20866/20866 [==============================] - 4943s 237ms/step
# - loss: 0.0412 - acc: 0.9902 - val_loss: 0.0094 - val_acc: 0.9982
def getModel1():
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

# 20866/20866 [==============================] - 4601s 221ms/step
# - loss: 0.0090 - acc: 0.9971 - val_loss: 0.0102 - val_acc: 0.9979
def getModel2():
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

# Made Dense layer have 64 nodes rather than 16
# 20866/20866 [==============================] - 4142s 199ms/step
# - loss: 0.0120 - acc: 0.9969 - val_loss: 0.0182 - val_acc: 0.9977
def getModel3():
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

# chnaged the learning rate to 0.00008 from 0.0001
# 20866/20866 [==============================] - 4855s 233ms/step
# - loss: 0.0800 - acc: 0.9843 - val_loss: 0.0313 - val_acc: 0.9964
def getModel4():
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

def getTrainingDataGenerator():
    return ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

def getTrainingGenerator():
    return getTrainingDataGenerator().flow_from_directory(
        trainingDataDir,
        target_size=(imgWidth, imgHeight),
        batch_size=batch_size,
        class_mode="binary")

def getValidationGenerator():
    validationDataGenerator = ImageDataGenerator(rescale=1./255)
    return validationDataGenerator.flow_from_directory(
        validationDataDir,
        target_size=(imgWidth, imgHeight),
        batch_size=batch_size,
        class_mode="binary")

epochs = 1
model_nr = "4"
model = getModel4()

trainingDataGenerator = getTrainingDataGenerator()
trainingGenerator = getTrainingGenerator()
validationGenerator = getValidationGenerator()

model.fit_generator(
    trainingGenerator,
    steps_per_epoch=len(trainingGenerator.filenames),
    epochs=epochs,
    validation_data=validationGenerator,
    validation_steps=len(validationGenerator.filenames),
    callbacks=[CSVLogger(trainingLogsPath,
                                            append=False,
                                            separator=";")], verbose=1)

# modelTrainedWeightsFile = modelPathPrefix + model_nr + "_nr_epochs_" + str(epochs) + modelPathSuffix
modelTrainedWeightsFile = "../trainedModels/model4.h5"
model.save_weights(modelTrainedWeightsFile)
print("model file: " + modelTrainedWeightsFile)
try:
    print("model: " + modelName)
except:
    print("error")
