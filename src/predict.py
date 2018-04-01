import pandas as pd
import os
from os.path import join
from config import Config
import model as modellib
import cv2
import utils
import json
import argparse

import visualize


model_path = './experiments/attempt4/furniture20180317T0042/mask_rcnn_furniture_0003.h5'

class_names = [
    'BG',
    'dining_table',
    'shelf',
    'armchair',
    'lamp',
    'couch',
    'desk',
    'chair',
    'bed',
    'wardrobe'
]


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'experiments', 'attempt4')


class MyConfig(Config):
    NAME = "furniture"

    IMAGES_PER_GPU = 2
    
    IMAGE_MIN_DIM = 576
    IMAGE_MAX_DIM = 768

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = len(class_names)
    
    
class InferenceConfig(MyConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

    
inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

    
def my_resize(image, config):
    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING
    )
    return image, window, scale, padding
    
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--img', required=True,
        help='Image path')
    parser.add_argument(
        '-o', '--out', default='predict.png',
        help='Image path')
    parser.add_argument(
        '-m', '--model', default=model_path,
        help='Model path')

    args = parser.parse_args()
    return args
    
    
if __name__ == "__main__":
    
    args = parse_arguments()
    
    model.load_weights(
        args.model,
        by_name=True
    )
    
    img = cv2.imread(args.img)[:, :, ::-1]
    img_copy = img.copy()
    h, w = img.shape[:2]
    img, window, scale, padding = my_resize(img, inference_config)
    results = model.detect([img], verbose=1)

    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'], save_path=args.out)
