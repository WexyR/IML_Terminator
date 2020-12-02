import numpy as np
from resources import GENERATED_DATASET_PATH

db = np.load(GENERATED_DATASET_PATH)
_db = dict()
img = np.array(db['img'], copy=True, dtype=np.uint8)
sensors = np.array(db['sensors'], dtype=np.float16)
cubeRGB = np.array(db['cubeRGB'], dtype=np.uint8)
cubeSize = np.array(db['cubeSize'], dtype=np.float16)
class_id = np.array(db['class_id'], dtype=np.uint8)

i = 0
index = 0
size=2000
while index < len(img):
    i+=1
    new_index = min(index+size, len(img))
    np.savez(GENERATED_DATASET_PATH.split('.npz')[0]+f"_{index}-{new_index-1}.npz",
             img=img[index:new_index],
             sensors=sensors[index:new_index],
             cubeRGB=cubeRGB[index:new_index],
             cubeSize=cubeSize[index:new_index],
             class_id=class_id[index:new_index],
             labels=db['labels'])
    index=new_index

print('done compressing sample dataset')