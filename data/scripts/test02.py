import cv2
import os
import numpy as np
from tqdm import tqdm
 
img_path = "/home/aizhiqing/aizhiqing/projects/python_projects/yolact/demo/test_yolo"
txt_path = "/home/aizhiqing/aizhiqing/projects/python_projects/yolact/demo/test_yolo"

vis_save_path = '/home/aizhiqing/aizhiqing/projects/python_projects/yolact/demo/test_yolo_vis'
if not os.path.exists(vis_save_path):
    os.makedirs(vis_save_path)
 
imgs = os.listdir(img_path)
suffix_list = ['.jpg', '.png', ]
imgs = [item for item in imgs if os.path.splitext(item)[-1].lower() in suffix_list]
txts = os.listdir(txt_path)
txts = [item for item in txts if item.endswith('.txt')]

for img in tqdm(imgs):
    image = cv2.imread(os.path.join(img_path, img))
    height, width, _ = image.shape
    
    prefix, _ = os.path.splitext(img)
    txt_file = prefix+ '.txt'
    if txt_file not in txts:
        continue
    file_handle = open(os.path.join(txt_path, txt_file))
    cnt_info = file_handle.readlines()
    # new_cnt_info = [line_str.replace("\n", "").split(" ") for line_str in cnt_info]
    new_cnt_info = [item.strip().split() for item in cnt_info]

    
    color_map = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for new_info in new_cnt_info:
        s = []
        for i in range(1, len(new_info), 2):
            b = [float(tmp) for tmp in new_info[i:i + 2]]
            s.append([int(b[0] * width), int(b[1] * height)])
        class_ = new_info[0]
        index = int(class_)
        cv2.polylines(image, [np.array(s, np.int32)], True, color_map[index], thickness = 3)
    
    # cv2.imshow('img2', img)
    # cv2.waitKey()
    cv2.imwrite(os.path.join(vis_save_path, img), image)