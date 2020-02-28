### Jared Homer, Alex Stephens
###############################################################################
import glob
import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models, utils

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

INPUT_IMG_SIZE = (28, 28)

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
        shutil.copyfile(src, dst)

    fnames_dst = ["{}.{}.jpg".format(class_name, i) for i in range(100, 200)]
    for i, fname in enumerate(fnames_dst):
        src = os.path.join(src_path, fnames_src[i + 100])
        dst = os.path.join(test_path, fname)
        shutil.copyfile(src, dst)

    size = len(fnames_src)
    fnames_dst = ["{}.{}.jpg".format(class_name, i) for i in range(200, size)]
    for i, fname in enumerate(fnames_dst):
        src = os.path.join(src_path, fnames_src[i + 200])
        dst = os.path.join(training_path, fname)
        shutil.copyfile(src, dst)

def resize_images(src_path, new_size):
    if not isinstance(src_path, str):
        raise Exception("src_path must be of type string.")
    if isinstance(new_size, tuple):
        if len(new_size) != 2:
            raise Exception("Given size is not a valid size.")
    else:
        raise Exception("Given size must be a tuple.")

    os.chdir(src_path)
    for fname in os.listdir(src_path):
        if fname.endswith(".jpg"):
            file = os.path.join(src_path, fname)
            src_img = cv2.imread(file)
            new_img = np.copy(src_img)
            new_img = cv2.resize(new_img, new_size)
            cv2.imwrite(fname, new_img)
    os.chdir(BASE_PATH)

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

###############################################################################

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

###############################################################################

