import pandas as pd
import os
from os.path import join
from config import Config
import model as modellib
import cv2
import utils
import json


test_data_dirpath = '/data/images_test'
model_path = './experiments/attempt4/furniture20180317T0042/mask_rcnn_furniture_0003.h5'
test_data = pd.read_csv('/data/sample_submission.csv')
submit_path = 'submit.csv'

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


def makedirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
        
id2img_name = dict()
for i, raw in test_data.iterrows():
    id2img_name[raw.id] = os.path.basename(json.loads(raw.input)['image'])


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'experiments', 'attempt4')
makedirs(MODEL_DIR)


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
    DETECTION_MIN_CONFIDENCE = 0.1
    DETECTION_NMS_THRESHOLD = 0.3

    
inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

model.load_weights(
    model_path,
    by_name=True
)
    
    
def my_resize(image, config):
    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING
    )
    return image, window, scale, padding
    
    
for k, v in list(test_data.iterrows()):
    id = int(v['id'])
    img_filepath = join(test_data_dirpath, id2img_name[id])
    if k % 100 == 0:
        print(k)
    if not os.path.exists(img_filepath):
        print('No image {} {}'.format(id, img_filepath))
        continue
    else:
        img = cv2.imread(img_filepath)[:, :, ::-1]
        img_copy = img.copy()
        h, w = img.shape[:2]
        img, window, scale, padding = my_resize(img, inference_config)
        
        results = model.detect([img], verbose=0)

        r = results[0]
        
        n = len(r['scores'])
        result_json = {cls: [] for cls in class_names[1:]}
        for i in range(n):
            class_id, roi, score = (
                r['class_ids'][i],
                r['rois'][i],
                r['scores'][i]
            )
            if score < inference_config.DETECTION_MIN_CONFIDENCE:
                continue
            cls = class_names[class_id]
            top_pad, left_pad = window[:2]
            def translate(x, y):
                x = ((x - left_pad) / scale) / w
                y = ((y - top_pad) / scale) / h
                return x, y
            top, left, bottom, right = roi
            left, top = translate(left, top)
            right, bottom = translate(right, bottom)
            
            rect = [
                [left, top],
                [left, bottom],
                [right, bottom],
                [right, top],
                [left, top]
            ]
            
            result_json[cls].append((rect, float(score)))
            
        result_json = {'aabb': result_json}
        test_data.set_value(k, 'prediction', json.dumps(result_json))
        
test_data.to_csv(submit_path, index=None)
