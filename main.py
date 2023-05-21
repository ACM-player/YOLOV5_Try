import cv2
import numpy as np
import datetime
import os
import csv
import time
# iing
import sys
from tqdm import  tqdm

from FaceDetector import yolov5,FaceFeat
from sklearn.metrics.pairwise import cosine_similarity

def cos_sim(vec1,vec2):

    vec1 = np.array(vec1)[0]
    vec2 = np.array(vec2)[0]
    cos_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))

    return cos_sim





names = ['face','']
det = yolov5('pretrained/best.onnx',names)
rec = FaceFeat('arcface.onnx')


fts = []
lbs = []

files = os.listdir('Images/data2/labeled_data_sample')

phs = open('list.txt','w')
for bs in tqdm(files):

    us = os.listdir('Images/data2/labeled_data_sample/'+str(bs))
    for u in us:
        lbs.append(float(bs))
        pwd = os.getcwd()
        u = pwd + '/Images/data2/labeled_data_sample/' +str(bs) +'/'+ u
        print(u)

        phs.write(u +'\n')
        img = cv2.imread(u)
        u = det.forward(img)

        x1, y1, x2, y2, name, cf =u[0]
        face = img[y1:y2, x1:x2, :]
        ft = rec([face])

        fts.append(ft)




fts = np.concatenate([fts],0)

lbs = np.concatenate([lbs],0)

np.save('features.npy',fts)

np.save('lbs.npy',lbs)
print(ft.shape)
