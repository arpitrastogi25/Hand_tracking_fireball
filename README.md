# Hand_tracking_fireball
Created a simple code for Hand Tracking application that genrates fireball illusion (like that in Dragon Ball Z) using opencv-python. This uses primary webcam to capture live video. And also a gif of a fireball is used.

1. This uses Mediapipe framework to detect the hands from the live webcam feed.

2. When the two hand's midpoint are apart by less than 'hand_distance_threshold' a small fireball is displayed, whose size and position are vary depending on the distance between the two hands and the mid-point between those two.

#------------------------------------------------------------------------------------------------------

Dependencies:-

import cv2;
import mediapipe as mp;
import numpy as np;
import time;
import math;
from PIL import Image;
