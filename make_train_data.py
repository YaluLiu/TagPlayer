import numpy as np 
import json
import os

def get_all_json(dir_path):
    file_lst = os.listdir(dir_path)
    file_lst = sorted(file_lst,  key=lambda x: os.path.getmtime(os.path.join(dir_path, x)))
    fns = []
    for fn in file_lst:
        if fn.endswith(".json"): 
            fn = os.path.join(dir_path,fn)       
            fns.append(fn)
    return fns

def preprocess(ann):
    kpts = ann['keypoints']
    bbox = ann['box']
    
    kpts = np.array(kpts).reshape(-1,3)[:,:2]
    cpts = (kpts[11] + kpts[12]) / 2
    kpts = kpts - cpts

    x, y, w, h = bbox
    kpts[:,0] = kpts[:,0]/w
    kpts[:,1] = kpts[:,1]/h
    
    return kpts.reshape(-1)

'''
other 0
climb 1
throw 2
aim   3
handgun 4
attack 5
['other','climb','throw','aim','handgun','attack']
'''

'''
stand 0
hands_up 1
raise_left 2
raise_right 3
touch_head 4
sit 5
['stand','hands_up','raise_left','raise_right','touch_head','sit']
'''


def parse_json(fn):
    tags = ['stand','hands_up','raise_left','raise_right','touch_head','sit']
    with open(fn,'r',encoding='utf-8') as tmp:
        persons = json.load(tmp)
        # for person in persons:
        #     for label_id,label in enumerate(tags):
        #         if label in person.keys() and person[label]:
        #             person["label"] = label_id
    return persons

def parse_kps(kps):
    kps = np.array(kps,dtype = np.int64)
    kps = kps.reshape(-1,3)[:,:2].reshape(-1)
    return kps

if __name__ == '__main__':
    fns  = get_all_json("image/demo/hands_up")
    fns += get_all_json("image/demo/raise_left")
    fns += get_all_json("image/demo/raise_right")
    fns += get_all_json("image/demo/sit")
    fns += get_all_json("image/demo/stand")
    fns += get_all_json("image/demo/touch_head")

    kps_data = []
    labels = []
    for fn in  fns:
        persons = parse_json(fn)
        for person in persons:
            kps = preprocess(person)
            kps_data.append(kps)
            labels.append(person['label'])
    kps_data = np.array(kps_data,dtype = np.float64)
    data_num = len(labels)
    label_data = np.zeros((data_num,6),dtype=np.int64)
    for idx in range(len(labels)):
        tmp_lable = labels[idx]
        label_data[idx][tmp_lable] = 1

    print(kps_data.shape)
    print(label_data.shape)
    np.save("feature_demo.npy", kps_data)
    np.save("label_demo.npy", label_data)



