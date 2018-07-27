import cv2
from darkflow.net.build import TFNet
import requests
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

url = "http://192.168.2.23:8080/shot.jpg"

while True:
    stime = time.time()
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
    img = cv2.imdecode(img_arr, -1)
    frame = cv2.flip(img, 1)
    results = tfnet.return_predict(frame)

    #if ret:
    for color, result in zip(colors,results):
        tl = (result['topleft']['x'],result['topleft']['y'])
        br = (result['bottomright']['x'],result['bottomright']['y'])
        label = result['label']
        confidence = result['confidence']
        text = '{}: {:.0f}%'.format(label,confidence * 100)
        frame = cv2.rectangle(frame,tl,br,color,5)
        frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX,
                            1,(0,0,0),2)
        cv2.imshow('frame',frame)
        print('FPS {:.1f}'.format(1 /(time.time() - stime)))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
