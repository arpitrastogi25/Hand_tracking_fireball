
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from PIL import Image


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        nHands = 0
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                nHands = nHands + 1
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img,nHands

    def findPosition(self,img, handNo=0, draw=True, pts=[]):
        h, w, c = img.shape
        lmList = []
        nHands = 0
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                for i in range(len(pts)):
                    if(pts[i]==id):
                        lmList.append([id, cx, cy])
                        if draw:
                            cv2.circle(img,(cx,cy),10,(0,0,0),cv2.FILLED)
        return lmList


def findmask(img):
    lower = np.array([0, 0, 0])
    upper = np.array([100, 100, 150])
    mask = cv2.inRange(img, lower, upper)
    return mask

def main():
    pTime = 0
    cTime = 0
    frames = 0
    cap = cv2.VideoCapture(0)
    cap.set(3,720)
    cap.set(4,480)
    detector = handDetector(detectionCon=0.7)
    start = 0
    hand_distance_threshold = 200
    imageObject = Image.open("ball.gif")
    no_frames = imageObject.n_frames
    ball_images = []
    for frame in range(0,no_frames):
        imageObject.seek(frame)
        tmp = imageObject.convert()
        img = cv2.cvtColor(np.asarray(tmp),cv2.COLOR_RGB2BGR)
        ball_images.append(img)
    ball_images = np.asarray(ball_images)
    gif_count = 0
    count = 0
    while True:
        success, img = cap.read()
        img = cv2.flip(img, flipCode=1)
        img,nHands = detector.findHands(img,draw=True)
        lmList = []
        for nhand in range(nHands):
            lmList.append(detector.findPosition(img,handNo=nhand, draw=True, pts=[9]))
            if(len(lmList)!=0):
                pass
                #print('hand no-',nhand,lmList[nhand])


        if nHands == 1:
            start = 0

        if nHands == 2:
            len_b_index_x1,len_b_index_y1 = lmList[0][0][1],lmList[0][0][2]
            len_b_index_x2,len_b_index_y2 = lmList[1][0][1],lmList[1][0][2]
            len_b_index_x = int(math.fabs(len_b_index_x2-len_b_index_x1))
            len_b_index_y = int(math.fabs(len_b_index_y2-len_b_index_y1))
            len_b_index = math.hypot(len_b_index_x, len_b_index_y)
            if(len_b_index < hand_distance_threshold):
                start = 1
                gif_count = 0

            if(start == 1):
                ball = ball_images[gif_count]
                cx = (len_b_index_x2 +len_b_index_x1)//2
                cy = (len_b_index_y2 +len_b_index_y1)//2
                ball_size = min(max(2,int(len_b_index_x//1),int(len_b_index_y//1)),ball.shape[0]*2)
                x_ball = max(0,cx-ball_size//2)
                y_ball = max(0,cy-ball_size//2)
                ball_temp_size = min(ball_size // 1, img.shape[0] - y_ball, img.shape[1])
                ball_temp = cv2.resize(ball,(ball_temp_size,ball_temp_size))
                mask = findmask(ball_temp)
                mask_inv = cv2.bitwise_not(mask)
                img_temp = img[y_ball:y_ball+mask.shape[0],x_ball:x_ball+mask.shape[1]]
                img_mask = cv2.bitwise_and(img_temp, img_temp, mask=mask)
                ball_mask = cv2.bitwise_and(ball_temp, ball_temp, mask=mask_inv)
                img_final = cv2.add(img_mask, ball_mask)

                img[y_ball:y_ball+mask.shape[0], x_ball:x_ball +mask.shape[1]] = img_final
                if(gif_count == 63):
                    gif_count = 0
                else:
                    gif_count = gif_count+1

        frames = frames + 1
        cTime = time.time()
        if (cTime - pTime > 1):
            fps = frames
            frames = 0
            pTime = cTime


        cv2.putText(img, 'fps-' + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        cv2.imshow('Image', img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

if __name__ == "__main__":
    main()