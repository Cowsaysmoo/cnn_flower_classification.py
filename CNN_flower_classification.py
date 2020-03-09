### Jared Homer, Alex Stephens
###############################################################################
# uncomment to run on Alex's PC for AMD GPU
#import plaidml.keras
#plaidml.keras.install_backend()

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
#from keras import backend as K # uncomment to run on Alex's PC for AMD GPU
from keras import layers, models, applications, optimizers
from keras.preprocessing import image

###############################################################################

BASE_PATH = os.getcwd()

# paths to original data
DATA_PATH       = os.path.join(BASE_PATH, "flower_photos")
DATA_DAISY_PATH = os.path.join(DATA_PATH, "daisy")
DATA_DANDE_PATH = os.path.join(DATA_PATH, "dandelion")
DATA_ROSE_PATH  = os.path.join(DATA_PATH, "roses")
DATA_SUN_PATH   = os.path.join(DATA_PATH, "sunflowers")
DATA_TULIP_PATH = os.path.join(DATA_PATH, "tulips")

DATA_PATHS = [
    DATA_DAISY_PATH,
    DATA_DANDE_PATH,
    DATA_ROSE_PATH,
    DATA_SUN_PATH,
    DATA_TULIP_PATH
]

CLASSES = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

# paths for training set
TRAIN_PATH       = os.path.join(BASE_PATH, "training_set")
TRAIN_DAISY_PATH = os.path.join(TRAIN_PATH, "daisy")
TRAIN_DANDE_PATH = os.path.join(TRAIN_PATH, "dandelion")
TRAIN_ROSE_PATH  = os.path.join(TRAIN_PATH, "rose")
TRAIN_SUN_PATH   = os.path.join(TRAIN_PATH, "sunflower")
TRAIN_TULIP_PATH = os.path.join(TRAIN_PATH, "tulip")

# paths for validation set
VAL_PATH        = os.path.join(BASE_PATH, "validation_set")
VAL_DAISY_PATH  = os.path.join(VAL_PATH, "daisy")
VAL_DANDE_PATH  = os.path.join(VAL_PATH, "dandelion")
VAL_ROSE_PATH   = os.path.join(VAL_PATH, "rose")
VAL_SUN_PATH    = os.path.join(VAL_PATH, "sunflower")
VAL_TULIP_PATH  = os.path.join(VAL_PATH, "tulip")

# paths for test set
TEST_PATH       = os.path.join(BASE_PATH, "test_set")
TEST_DAISY_PATH = os.path.join(TEST_PATH, "daisy")
TEST_DANDE_PATH = os.path.join(TEST_PATH, "dandelion")
TEST_ROSE_PATH  = os.path.join(TEST_PATH, "rose")
TEST_SUN_PATH   = os.path.join(TEST_PATH, "sunflower")
TEST_TULIP_PATH = os.path.join(TEST_PATH, "tulip")

TRAIN_PATHS = [
    TRAIN_DAISY_PATH,
    TRAIN_DANDE_PATH,
    TRAIN_ROSE_PATH,
    TRAIN_SUN_PATH,
    TRAIN_TULIP_PATH
]

VAL_PATHS = [
    VAL_DAISY_PATH,
    VAL_DANDE_PATH,
    VAL_ROSE_PATH,
    VAL_SUN_PATH,
    VAL_TULIP_PATH
]

TEST_PATHS = [
    TEST_DAISY_PATH,
    TEST_DANDE_PATH,
    TEST_ROSE_PATH,
    TEST_SUN_PATH,
    TEST_TULIP_PATH
]

INPUT_SIZE = (150, 150)

###############################################################################

def path_checks(path_name):
    if not os.path.isdir(path_name):
        raise Exception("{} is not a directory.".format(path_name))
    if not os.listdir(path_name):
        raise Exception("{} is empty.".format(path_name))

def partition_sets(
        src_path, val_path, test_path,
        training_path, class_name):
    if not isinstance(class_name, str):
        msg = "Given class_name is of type {}, it should be of type string"
        raise Exception(msg.format(type(class_name)))

    fnames_src = glob.glob("{}/*.jpg".format(src_path))
    fnames_dst = ["{}.{}.jpg".format(class_name, i) for i in range(100)]
    for i, fname in enumerate(fnames_dst):
        src = os.path.join(src_path, fnames_src[i])
        dst = os.path.join(val_path, fname)
        if src != dst:
            shutil.copyfile(src, dst)

    fnames_dst = ["{}.{}.jpg".format(class_name, i) for i in range(100, 200)]
    for i, fname in enumerate(fnames_dst):
        src = os.path.join(src_path, fnames_src[i + 100])
        dst = os.path.join(test_path, fname)
        if src != dst:
            shutil.copyfile(src, dst)

    size = len(fnames_src)
    fnames_dst = ["{}.{}.jpg".format(class_name, i) for i in range(200, size)]
    for i, fname in enumerate(fnames_dst):
        src = os.path.join(src_path, fnames_src[i + 200])
        dst = os.path.join(training_path, fname)
        if src != dst:
            shutil.copyfile(src, dst)

###############################################################################

# Checking that data path was created correctly 
# by the user BEFORE running this file
if not os.path.isdir(DATA_PATH):
    raise Exception("{} is not a directory.".format(DATA_PATH))

path_checks(DATA_DAISY_PATH)
path_checks(DATA_DANDE_PATH)
path_checks(DATA_ROSE_PATH)
path_checks(DATA_SUN_PATH)
path_checks(DATA_TULIP_PATH)

###############################################################################

# Create folders for partitioned data
for pth in TRAIN_PATHS:
    Path(pth).mkdir(parents=True, exist_ok=True)
for pth in VAL_PATHS:
    Path(pth).mkdir(parents=True, exist_ok=True)
for pth in TEST_PATHS:
    Path(pth).mkdir(parents=True, exist_ok=True)

###############################################################################

print("Partitioning dataset, this could take a while...")
for i, pth in enumerate(DATA_PATHS):
    partition_sets(pth, VAL_PATHS[i], TEST_PATHS[i], TRAIN_PATHS[i], CLASSES[i])
print("Partitioning completed.")

###############################################################################

# import VGG model as base and add classification layers
conv_base = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(INPUT_SIZE[0],INPUT_SIZE[1],3))

network = models.Sequential()
network.add(conv_base)
network.add(layers.Flatten())
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dropout(0.1))
network.add(layers.Dense(5, activation='softmax'))

# unlock VGG's last block of layers, lock the rest
for layer in conv_base.layers[:-4]:
    layer.trainable = False

# data augmentation generators and normalization
train_datagen = image.ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)
val_datagen = image.ImageDataGenerator(rescale=1.0/255)
test_datagen = image.ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=INPUT_SIZE,
    batch_size=50,
    class_mode="categorical"
)
val_gen = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=INPUT_SIZE,
    batch_size=50,
    class_mode="categorical"
)
test_gen = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=INPUT_SIZE,
    batch_size=100, # lower this number if you get an out of memory error
    class_mode="categorical"
)

network.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# train model
history = network.fit_generator(
    train_gen,
    steps_per_epoch=train_gen.samples//train_gen.batch_size,
    epochs=10,
    validation_data=val_gen,
    validation_steps=val_gen.samples//val_gen.batch_size,
    verbose=1
)

# plotting network statistics
acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure(2)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# test model
print("Getting test accuracy, this could take a while.")
test_loss, test_acc = network.evaluate_generator(test_gen)
print("Test accuracy: {:.2f}%, or {}".format(test_acc*100, test_acc))