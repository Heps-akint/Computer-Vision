import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

while True:
    success, image = capture.read()
    cv2.imshow("Image", image)
    cv2.waitKey(1)
