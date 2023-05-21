from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer, QDate, Qt
from PyQt5.QtWidgets import QDialog,QMessageBox,QInputDialog,QLineEdit
import cv2
import numpy as np
import datetime
import os
import csv
import time
from PyQt5.QtWidgets import QApplication, QDialog,QMainWindow
import sys


from FaceDetector import yolov5,FaceFeat
from sklearn.metrics.pairwise import cosine_similarity

def cos_sim(vec1,vec2):

    vec1 = np.array(vec1)[0]
    vec2 = np.array(vec2)[0]
    cos_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))

    return cos_sim



class Ui_main(QDialog):
    def __init__(self):
        super(Ui_main, self).__init__()
        loadUi("./outputwindow.ui", self)   #加载输出窗体UI

        self.image = None


        self.name = 'unknown'
        self.setWindowTitle('')

        self.fts = []
        self.names = []
        self.det = None
        self.feat = None
        names = ['face','']
        self.det = yolov5('pretrained/best.onnx',names)
        self.rec = FaceFeat('arcface.onnx')
        self.startVideo('2.mp4')

        self.timer.timeout.connect(self.update_frame)  # 超时连接输出
        self.timer.start(0)

    @pyqtSlot()
    def startVideo(self, camera_name):
        if len(camera_name) == 1:
            self.capture = cv2.VideoCapture(int(camera_name))
        else:
            self.capture = cv2.VideoCapture(camera_name)

        self.timer = QTimer(self)
        path = 'Images'
        class_names = os.listdir(path)
        # print(self.class_names)

        class_names = [c for c in class_names if os.path.isdir(os.path.join(path,c))]


        self.encode_list = []

        self.class_names = []
        for cl in class_names:
            li = os.path.join(path,cl)

            images = os.listdir(li)

            for img in images:
                img = cv2.imread(os.path.join(li,img))

                self.class_names.append(cl)

                x1, y1, x2, y2, name, cf  = self.det.forward(img)[0]

                face = img[y1:y2, x1:x2, :]
                ft = self.rec([face])

                self.encode_list.append(ft)

    def face_rec_(self, frame):


        detr = self.det.forward(frame)

        if len(detr)>0:
            x1, y1, x2, y2, name, cf = detr[0]


            face = frame[y1:y2,x1:x2,:]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)



            ft = self.rec([face])

            sims = [cos_sim(ft,vec) for vec in self.encode_list]
            sims = np.array(sims)

            maxg = sims.argmax()
            score = sims[maxg]
            # print(sims,score)
            if score>0.6:

                self.name = self.class_names[maxg]

            else:
                self.name = 'unknown'
                
            cv2.putText(frame, self.name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            self.name = 'unknown'
        return frame


    def update_frame(self):

        ret, self.image = self.capture.read()
        t = time.time()
        self.displayImage(self.image, 1)


    def displayImage(self, image):

        image = cv2.resize(image, (640, 480))
        
        
        try:

            image = self.face_rec_(image)
        except Exception as e:
            print(e) 
        
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        

        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()


        self.label.setPixmap(QPixmap.fromImage(outImage))
        self.label.setScaledContents(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = Ui_main()
    ui.show()
    sys.exit(app.exec_())


