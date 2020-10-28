# -*- coding: UTF-8 -*-
import json
import requests
import threading
import time
import os
import cv2
import numpy as np

def get_file_list(file_path):
    dir_list = os.listdir(file_path)

    ret_list = []
    for file in dir_list:
        if file.endswith(".jpg"):
            file = os.path.join(file_path,file)
            ret_list.append(file)

    # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
    # os.path.getmtime() 函数是获取文件最后修改时间
    # os.path.getctime() 函数是获取文件最后创建时间
    ret_list = sorted(ret_list,  key=lambda x: os.path.getmtime(x))
    #ret_list = sorted(ret_list,  key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
    # print(dir_list)
    return ret_list

def get_dirs(root_path):
    dir_list = os.listdir(root_path)
    dirs = []
    for file in dir_list:
        file = os.path.join(root_path,file)
        if os.path.isdir(file):
            dirs.append(file)
    return dirs

#获取接口地址
def get_url_alphapose():
    #端口号
    port = 7008
    #主机地址
    host = "192.168.200.233"
    cam_id = 0
    api = "api_detect_climb"
    detect_url = "http://{}:{}/{}/{}".format(host,port, api, cam_id)
    return detect_url

#发送图片给检测服务器docker
def post_image(post_url,frame):
    frame_encoded = cv2.imencode(".jpg", frame)[1]
    Send_file = {'image': frame_encoded.tostring()}
    jsondata = requests.post(post_url,files=Send_file)

    #output
    if jsondata.status_code == requests.codes.ok:
        # print(jsondata.json())
        return jsondata.json()
    else:
        print("Error")

def resize_frame(frame):
    height,width,_ = frame.shape
    size = (int(width*0.5), int(height*0.5))
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame

def show_rectangle(img,rec):
    x0,y0,w,h = np.array(rec,dtype=np.int32)
    print(x0,y0,w,h)
    cv2.rectangle(img,(x0,y0),(x0+w,y0+h),color = (255,255,0),thickness=2)

def show_rectangle(frame,data):
    for person in data:
        frame_show = np.copy(frame)
        person.pop('climb', None)
        x0,y0,w,h = np.array(person["box"],dtype=np.int32)
        cv2.rectangle(frame_show,(x0,y0),(x0+w,y0+h),color = (255,255,0),thickness=2)
        cv2.imshow("aa",frame_show)
        key = cv2.waitKey(0)
        person["attack"] = True
        if key == ord('a'):
            person["attack"] = False
        print(person["attack"])

def make_tag(frame,data,tag):
    for person in data:
        person.pop('climb', None)
        tags = ['stand','hands_up','raise_left','raise_right','touch_head','sit']
        for label_id,label in enumerate(tags):
            if label == tag:
                person["label"] = label_id
    
def save_data(data,file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)
    print("save {} success!".format(file_path))

if __name__ == '__main__':
    url = get_url_alphapose()
    # dir_path = "D:/action_image/attack/one_attack"
    # dir_path = "D:/action_image/throw/throw"
    # dir_path = "D:/action_image/throw/other"
    # dir_path = "D:/action_image/cross/cross"
    root_dir_path = "D:/action_image/image/demo"
    dirs = get_dirs(root_dir_path)
    for dir_path in dirs:
        tag = dir_path.split("\\")[1]
        print(dir_path,tag)
        print("="*10)
        files = get_file_list(dir_path)

        for idx,file in enumerate(files):
            print(idx,":",file, end = ", ")
            json_file = file.replace('.jpg',".json")
            # json_file = os.path.join(dir_path,json_file)
            # file = os.path.join(dir_path,file)
            frame = cv2.imread(file)
            data = post_image(url,frame)
            show_rectangle(frame,data)
            break
            # make_tag(frame,data,tag)
            # save_data(data,json_file)
        break

            
            


