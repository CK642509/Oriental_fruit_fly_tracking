import cv2
import numpy as np
import pandas as pd
import math

from datetime import timedelta



video = cv2.VideoCapture(r'C:\Users\CK642509\Desktop\文件\Python\track fly\5.mp4')
fps = video.get(cv2.CAP_PROP_FPS)   # fps = 30


object_detector = cv2.createBackgroundSubtractorMOG2(history=100000, varThreshold=40)

position = []
frame_total_top = 0
frame_total_bot = 0
flag = 1   # top = 1, bot = -1, left = 0

while True:
    ret, frame = video.read()   # if frame is read correctly ret is True
    frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)   # resize
    height, width, _ = frame.shape
    # print(height, width)   # 540, 960

    # ROI
    roi = frame[:,:700]
    
    # object detection
    # mask = object_detector.apply(frame)
    test = object_detector.apply(roi)
    ret, mask = cv2.threshold(test, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)


    countours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            position_now = [int(math.floor(x+0.5*w)),int(math.floor(y+0.5*h))]
            # print(position_now)

            cv2.rectangle(roi, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(roi, (int(math.floor(x+0.5*w)), int(math.floor(y+0.5*h))), 3, (0,0,255), -1)

            # cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)

            if len(position) == 0:
                # position.append(position_now)
                position.append([648,125])   # start position
            else:
                distance = math.sqrt((position_now[0]-position[-1][0])**2 + (position_now[1]-position[-1][1])**2)
                if distance < 50:
                    position.append(position_now)
            
    arr = np.array(position, dtype=np.int32)
    cv2.polylines(roi, [arr], False, (255,0,0))


    # determine postion
    if len(position) > 0:
        pos_x = position[-1][0]
        pos_y = position[-1][1]

        if pos_x >= 380 and pos_x <= 700:
            if pos_y >= 50 and pos_y <= 300:
                flag = 1
            elif pos_y >= 301 and pos_y <= 540:
                flag = -1
            else:
                flag = 0
        else:
            flag = 0

    if flag == 1:
        frame_total_top += 1
    elif flag == -1:
        frame_total_bot += 1

    # count time
    time_top = timedelta(seconds=(frame_total_top/fps))
    time_bot = timedelta(seconds=(frame_total_bot/fps))

    # show time
    cv2.rectangle(frame, [380, 50], [700, 300], (0,0,255),2)
    cv2.rectangle(frame, [380, 301], [700, 540], (0,0,255),2)
    cv2.putText(frame, "{}".format(time_top)[:10], [710, 290], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
    cv2.putText(frame, "{}".format(time_bot)[:10], [710, 330], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)

    test_3d = np.stack([test, test, test], axis=-1)
    # cv2.imshow("Test3d", test_3d)
    # cv2.imshow("Test", test)   # 2
    # cv2.imshow("ROI", roi)   # 3
    # cv2.imshow("Frame", frame)   # 3
    # cv2.imshow("Mask", mask)   # 2

    combine = np.concatenate((frame, test_3d), axis=1)
    combine = cv2.resize(combine, None, None, fx=0.8, fy=0.8)
    cv2.imshow("All", combine)

    

    key = cv2.waitKey(30)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows
