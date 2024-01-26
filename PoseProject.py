import cv2
import mediapipe as mp
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()

while True:
    success, image = cap.read()
    image = detector.findPose(image)
    lmList = detector.findPosition(image)
    if len(lmList) !=0:
        print(lmList)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    
    cv2.imshow("Image", image)
    cv2.waitKey(1)