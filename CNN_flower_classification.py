### Jared Homer, Alex Stephens
###############################################################################
# uncomment to run on Alex's PC for AMD GPU
# import plaidml.keras
# plaidml.keras.install_backend()

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# from keras import backend as K # uncomment to run on Alex's PC for AMD GPU
from keras import layers, models, utils
from keras.preprocessing import image

import cv2

###############################################################################

BASE_PATH = os.getcwd()

# paths to original data
DATA_PATH       = os.path.join(BASE_PATH, "flower_photos")
DATA_DAISY_PATH = os.path.join(DATA_PATH, "daisy")
DATA_DANDE_PATH = os.path.join(DATA_PATH, "dandelion")
DATA_ROSE_PATH  = os.path.join(DATA_PATH, "roses")
DATA_SUN_PATH   = os.path.join(DATA_PATH, "sunflowers")
DATA_TULIP_PATH = os.path.join(DATA_PATH, "tulips")

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

LABELS = [
    [1, 0, 0, 0, 0], # Daisy
    [0, 1, 0, 0, 0], # Dandelion
    [0, 0, 1, 0, 0], # Rose
    [0, 0, 0, 1, 0], # Sun
    [0, 0, 0, 0, 1], # Tulip
]

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

INPUT_IMG_SIZE = (100, 100)

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

def resize_images(src_path, new_size):
    if not isinstance(src_path, str):
        raise Exception("src_path must be of type string.")
    if isinstance(new_size, tuple):
        if len(new_size) != 2:
            raise Exception("Given size is not a valid size.")
    else:
        raise Exception("Given size must be a tuple.")

    fnames = glob.glob("{}/*.jpg".format(src_path))
    for fname in fnames:
        file = os.path.join(src_path, fname)
        src_img = cv2.imread(file)
        if src_img.shape != new_size:
            new_img = np.copy(src_img)
            new_img = cv2.resize(new_img, new_size)
            os.chdir(src_path)
            cv2.imwrite(fname, new_img)
            os.chdir(BASE_PATH)

def add_imgs_labels(src_path, src_imgs, src_labels, label):
    if not isinstance(src_path, str):
        raise Exception("src_path must be a list.")
    if not isinstance(src_imgs, list):
        raise Exception("src_imgs must be a list.")
    if not isinstance(src_labels, list):
        raise Exception("src_labels must be a list.")

    fnames = glob.glob("{}/*.jpg".format(src_path))
    for fname in fnames:
        img = cv2.imread(fname)
        src_imgs.append(img)
        src_labels.append(label)
    return src_imgs, src_labels

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
Path(TRAIN_DAISY_PATH).mkdir(parents=True, exist_ok=True)
Path(TRAIN_DANDE_PATH).mkdir(parents=True, exist_ok=True)
Path(TRAIN_ROSE_PATH).mkdir(parents=True, exist_ok=True)
Path(TRAIN_SUN_PATH).mkdir(parents=True, exist_ok=True)
Path(TRAIN_TULIP_PATH).mkdir(parents=True, exist_ok=True)
Path(VAL_DAISY_PATH).mkdir(parents=True, exist_ok=True)
Path(VAL_DANDE_PATH).mkdir(parents=True, exist_ok=True)
Path(VAL_ROSE_PATH).mkdir(parents=True, exist_ok=True)
Path(VAL_SUN_PATH).mkdir(parents=True, exist_ok=True)
Path(VAL_TULIP_PATH).mkdir(parents=True, exist_ok=True)
Path(TEST_DAISY_PATH).mkdir(parents=True, exist_ok=True)
Path(TEST_DANDE_PATH).mkdir(parents=True, exist_ok=True)
Path(TEST_ROSE_PATH).mkdir(parents=True, exist_ok=True)
Path(TEST_SUN_PATH).mkdir(parents=True, exist_ok=True)
Path(TEST_TULIP_PATH).mkdir(parents=True, exist_ok=True)

###############################################################################

print("Partitioning dataset, this could take a while...")
partition_sets(DATA_DAISY_PATH, VAL_DAISY_PATH, 
               TEST_DAISY_PATH, TRAIN_DAISY_PATH, "daisy")
partition_sets(DATA_DANDE_PATH, VAL_DANDE_PATH,
               TEST_DANDE_PATH, TRAIN_DANDE_PATH, "dandelion")
partition_sets(DATA_ROSE_PATH, VAL_ROSE_PATH,
               TEST_ROSE_PATH, TRAIN_ROSE_PATH, "rose")
partition_sets(DATA_SUN_PATH, VAL_SUN_PATH,
               TEST_SUN_PATH, TRAIN_SUN_PATH, "sunflower")
partition_sets(DATA_TULIP_PATH, VAL_TULIP_PATH,
               TEST_TULIP_PATH, TRAIN_TULIP_PATH, "tulip")
print("Partitioning completed.")

###############################################################################

print("Resizing images, this could take a while...")
resize_images(VAL_DAISY_PATH, INPUT_IMG_SIZE)
resize_images(VAL_DANDE_PATH, INPUT_IMG_SIZE)
resize_images(VAL_ROSE_PATH, INPUT_IMG_SIZE)
resize_images(VAL_SUN_PATH, INPUT_IMG_SIZE)
resize_images(VAL_TULIP_PATH, INPUT_IMG_SIZE)
resize_images(TEST_DAISY_PATH, INPUT_IMG_SIZE)
resize_images(TEST_DANDE_PATH, INPUT_IMG_SIZE)
resize_images(TEST_ROSE_PATH, INPUT_IMG_SIZE)
resize_images(TEST_SUN_PATH, INPUT_IMG_SIZE)
resize_images(TEST_TULIP_PATH, INPUT_IMG_SIZE)
resize_images(TRAIN_DAISY_PATH, INPUT_IMG_SIZE)
resize_images(TRAIN_DANDE_PATH, INPUT_IMG_SIZE)
resize_images(TRAIN_ROSE_PATH, INPUT_IMG_SIZE)
resize_images(TRAIN_SUN_PATH, INPUT_IMG_SIZE)
resize_images(TRAIN_TULIP_PATH, INPUT_IMG_SIZE)
print("Resizing completed.")

###############################################################################

# prepare training image set and labels
train_images = []
train_labels = []

for i, path in enumerate(TRAIN_PATHS):
    train_images, train_labels = add_imgs_labels(path, train_images,
                                                 train_labels, LABELS[i])

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# prepare validation image set and labels
val_images = []
val_labels = []

for i, path in enumerate(VAL_PATHS):
    val_images, val_labels = add_imgs_labels(path, val_images,
                                             val_labels, LABELS[i])

val_images = np.array(val_images)
val_labels = np.array(val_labels)

# prepare test image set and labels
test_images = []
test_labels = []

for i, path in enumerate(TEST_PATHS):
    test_images, test_labels = add_imgs_labels(path, test_images,
                                               test_labels, LABELS[i])

test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_datagen = image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)
val_datagen = image.ImageDataGenerator()
test_datagen = image.ImageDataGenerator()

# normalize
train_images = train_images.astype('float32') / 255
val_images = val_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_gen = train_datagen.flow((train_images, train_labels), batch_size=32)
val_gen = val_datagen.flow((val_images, val_labels), batch_size=500)
test_gen = test_datagen.flow((test_images, test_labels), batch_size=500)

# create network architecture
network = models.Sequential()

network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=train_images[0].shape))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.Flatten())
network.add(layers.Dropout(0.5))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dense(5, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# fit network
history=network.fit_generator(train_gen, steps_per_epoch=100, epochs=50,
                              validation_data=val_gen, validation_steps=50)

# plotting network statistics
acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'bo', label='Validation accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure(2)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation accuracy')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# test model
test_loss, test_acc = network.evaluate_generator(test_gen)
print(test_acc)