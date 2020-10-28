import json
import numpy as np
import cv2
import os
from joblib import load

def preprocess(ann):
    kpts = ann['keypoints']
    bbox = ann['box']
    
    kpts = np.array(kpts).reshape(-1,3)[:,:2]
    cpts = (kpts[11] + kpts[12]) / 2
    kpts = kpts - cpts

    x, y, w, h = bbox
    kpts[:,0] = kpts[:,0]/w
    kpts[:,1] = kpts[:,1]/h
    
    return kpts.reshape(1,-1)

def get_all_imgs(dir_path):
    file_lst = os.listdir(dir_path)
    file_lst = sorted(file_lst,  key=lambda x: os.path.getmtime(os.path.join(dir_path, x)))
    fns = []
    for fn in file_lst:
        if fn.endswith(".jpg"): 
            fn = os.path.join(dir_path,fn)
            fns.append(fn)
    return fns

if __name__ == '__main__':
    clf = load("action_model.joblib")
    # dir_path = "throw/throw"
    # imgs = get_all_imgs(dir_path)
    # for img_path in imgs:
    #     json_path = img_path.replace(".jpg",".json")
    #     frame = cv2.imread(img_path)
    #     cv2.imshow("frame",frame)
    #     cv2.waitKey(0)
    #     break


