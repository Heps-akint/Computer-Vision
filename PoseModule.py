import cv2
import mediapipe as mp
import time


class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackingCon=0.5):
        
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackingCon)

    def findPose(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return image
    
    def findPosition(self, image, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = image.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(image, (cx,cy), 5, (255,0,0), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()

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


    
if __name__ == "__main__":
    main()