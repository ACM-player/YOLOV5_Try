import torch

from PIL import Image
import onnxruntime as ort

import json
import cv2
import numpy as np
from utils import letterbox, non_max_suppression, scale_coords
import onnxruntime


class yolov5:
    def __init__(self, onnx_path, names):
        self.imgsz = 640
        self.conf_thres = 0.1
        self.iou_thres = 0.2
        self.max_det = 1000
        self.names = names
        self.shape = None
        sess = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.input_name = sess.get_inputs()[0].name
        output_names = []
        for i in range(len(sess.get_outputs())):
            print('output shape:', sess.get_outputs()[i].name)
            output_names.append(sess.get_outputs()[i].name)

        self.output_name = sess.get_outputs()[0].name
        self.sess = sess

    def preprocess(self, img):
        self.shape = img.shape
        img, _, _ = letterbox(img)
        img = np.transpose(img, (2, 0, 1))[::-1]

        img = np.expand_dims(img, axis=0).astype(np.float32) / 255
        img = img.copy()
        return img

    def postprocess(self, preds):
        # print(preds)
        pred = non_max_suppression(preds, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

        # print(pred)
        xyxys = []
        for i, det in enumerate(pred):  # detections per image
            if det != None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords((self.imgsz, self.imgsz), det[:, :4], self.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if c >= len(self.names):
                        continue
                    label = self.names[c]
                    # labels = labels + ' ' + label
                    xyxy = [int(a) for a in xyxy]
                    xyxys.append(xyxy)
                    xyxy.append(label)
                    xyxy.append(float(conf))

        return xyxys

    def forward(self, img):
        # img = cv2.imread(self.test_img)
        img0 = img.copy()
        img = self.preprocess(img)

        r = self.sess.run(None, {'images': img})
        # print(r[0].shape)

        rr = self.postprocess(r[0])

        res = []
        for i, u in enumerate(rr):
            u = [c for i, c in enumerate(u)]
            x1, y1, x2, y2, name, cf = u
            if len(name)==0:
                continue
            # img0 = self.drawPred(img0, name, cf, x1, y1, x2, y2)
            res.append(u)

        res.sort(key=lambda x:(x[2]-x[0])*(x[3]-x[1]),reverse=True)
        return res

    def drawPred(self, frame, name, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (name, label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame


class FaceFeat:
    def __init__(self,path):
        self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])

        kt = np.zeros([1,3,112, 112],dtype=np.float32)

        onnx_result = self.session.run([], input_feed={'x': kt})

    def preprocess(self,img):

        img = Image.fromarray(img).convert('RGB')
        img = np.array(img)
        img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        return img


    def __call__(self,crops):
        crops = [self.preprocess(c) for c in crops]
        batch = np.concatenate(crops,0)
        # print(batch.shape)
        vec = self.session.run([], input_feed={'x': batch})
        ft = vec[0]

        return ft

if __name__ == '__main__':
    # extrater = FaceFeatureExtractor('../weights/arcface.pth')
    # im1 = cv2.imread('1.jpg')
    im1 = np.ones([112,112,3])
    im1 = im1.astype(np.uint8)
    # ft = extrater([im1])
    extraterox = FaceFeat('arcface.onnx')
    ftx = extraterox([im1])

    # error = ft -ftx
    print(ftx.shape)

