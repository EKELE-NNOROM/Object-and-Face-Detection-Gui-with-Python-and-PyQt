import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': 'cfg/yolo9000.cfg',
    'load': 'bin/yolo9000.weights',
    'threshold': 0.2,
    }

net = TFNet(options)

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

while True:
    starttime = time.time()
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    results = net.return_predict(frame)

    if ret:
        for color, result in zip(colors,results):
            topleft = (result['topleft']['x'],result['topleft']['y'])
            botright = (result['bottomright']['x'],result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label,confidence * 100)
            frame = cv2.rectangle(frame,topleft,botright,color,5)
            frame = cv2.putText(frame, text, topleft, cv2.FONT_HERSHEY_COMPLEX,
                                1,(0,0,0),2)
            cv2.imshow('frame',frame)
            print('FPS {:.1f}'.format(1 /(time.time() - starttime)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
