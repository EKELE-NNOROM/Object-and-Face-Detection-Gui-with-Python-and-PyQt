import sys
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi


class FaceDetection(QDialog):
    def __init__(self):
        super(FaceDetection,self).__init__()
        loadUi('face_detection.ui',self)
        self.image=None
        self.processedImage=None
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.detectButton.setCheckable(True)
        self.detectButton.toggled.connect(self.detect_webcam_face)
        self.face_Enabled=False
        self.faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=self.faceCascade.detectMultiScale(gray,1.2,5,minSize=(90,90))

        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

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
    window=FaceDetection()
    window.setWindowTitle('Face Detection App')
    window.show()
    sys.exit(app.exec_())
