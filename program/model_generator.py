import os
import sys
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/matias/workspace-ia/Object-Detection-System-using-MaskR-CNN-Raspberry-Pi/")
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

import mrcnn.model as modellib
from mrcnn import utils
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)