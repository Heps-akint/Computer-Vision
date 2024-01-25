import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0 #previous time
cTime = 0 #current time
capture = cv2.VideoCapture(0)
detector = htm.handDetector()
while True:
    success, image = capture.read()
    image = detector.findHands(image)
    lmList = detector.findPosition(image)
    if len(lmList) !=0:
        print(lmList[4])
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime