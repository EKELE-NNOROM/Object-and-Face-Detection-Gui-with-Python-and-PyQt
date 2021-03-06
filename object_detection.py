import sys
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': 'cfg/yolov2.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.2,
    'gpu': 0.8
    }

tfnet = TFNet(options)

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

class ObjectDetection(QDialog):
    def __init__(self):
        super(ObjectDetection,self).__init__()
        loadUi('face_detection.ui',self)
        self.image=None
        self.processedImage=None
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.detectButton.setCheckable(True)
        self.detectButton.toggled.connect(self.detect_webcam_face)
        self.face_Enabled=False
        self.options={
                'model': 'cfg/yolov2-12.cfg',
                'load': 53904,
                'threshold': 0.1,
                'gpu': 0.8
                }


    def detect_webcam_face(self,status):
            if status:
                self.detectButton.setText('Stop Detection')
                self.face_Enabled=True
            else:
                self.detectButton.setText('Detect Face')
                self.face_Enabled=False

    def start_webcam(self):
        self.capture=cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret,self.image=self.capture.read()
        self.image=cv2.flip(self.image,1)

        self.displayImage(self.image, 1)
        if(self.face_Enabled):
            detected_image=self.detect_face(self.image)
            self.displayImage(detected_image,2)
        else:
            self.displayImage(self.image,1)


    def detect_face(self,img):
        results=tfnet.return_predict(img)

        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            img = cv2.rectangle(img, tl, br, color, 5)
            img = cv2.putText(img, text, tl, cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 0), 2)
        return img


    def stop_webcam(self):
        self.timer.stop()


    def displayImage(self,img,window=1):
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888

        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        #BGR>>RGB
        outImage=outImage.rgbSwapped()

        if window==1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)

        if window==2:
            self.processedImgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.processedImgLabel.setScaledContents(True)

if __name__=='__main__':
    app=QApplication(sys.argv)
    window=ObjectDetection()
    window.setWindowTitle('Object Detection App')
    window.show()
    sys.exit(app.exec_())
