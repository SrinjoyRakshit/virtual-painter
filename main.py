import cv2 as oc
import numpy as np
import mediapipe as mp
from collections import deque
 
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

kernel = np.ones((5,5),np,uint8)

colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255)]
colorIndex = 0

paintWindow = np.zeroes((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)

cv2.putText(paintWindow, "CLEAR", (49,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow, "YELOW", (520,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence= 0.7)
mpDraw = mp.solutions.drawings.utils 

cap = cv2.VideoCapture(0)
ret, frame_temp = cap.read()
while ret:
    ret, frame = cap.read()
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    #hsv = cv2.cvtColor(frame, cv2.COLOR.BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR.BGR2RGB)
    
    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
    frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
    frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), 2)
    cv2.putText(paintWindow, "CLEAR", (49,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(paintWindow, "YELOW", (520,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
    #frame = cv2.cvtColor(hsv, cv2.COLOR.HSV2BGR)

    result = hands.process(framergb)
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark: 
                # # print(id, lm)
                # print (lm.x)
                # print (lm.y)
                #Adjust according to frame size
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
        
                landmarks.append([lmx, lmy])
        
            #drawing landmarks on frame
            mpDraw.draw_landmarks(frame, handslms, mpHands.hAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        print (center[1]-thumb[1])
        if (thumb[1]-center[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index +=1
            gpoints.append(deque(maxlen=512))
            green_index +=1
            rpoints.append(deque(maxlen=512))
            red_index +=1
            ypoints.append(deque(maxlen=512))
            yellow_index +=1
    
        elif center[1] <= 65:
            if 40 <= center[0] <= 140: #Clear Button
                bpoints = [deque(maxlen = 512)]
                gpoints = [deque(maxlen = 512)]
                rpoints = [deque(maxlen = 512)]
                ypoints = [deque(maxlen = 512)]
            
                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0
        
                paintWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                colorIndex = 0 #Blue
            elif 275 <= center[0] <= 370:
                colorIndex = 1 #Green
            elif 390 <= center[0] <= 485:
                colorIndex = 2 #Red
            elif 505 <= center[0] <= 600:
                colorIndex = 3 #Yellow
        
        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        blue_index +=1
        gpoints.append(deque(maxlen=512))
        green_index +=1
        rpoints.append(deque(maxlen=512))
        red_index +=1
        ypoints.append(deque(maxlen=512))
        yellow_index +=1

    points = [bpoinst, gpoints, rpoints, ypoints]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k-1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, frame, points[i][j][k-1], points[i][j][k], colors[i], 2)
              
    cv2.imshow("output", frame)
    cv2.imshow("paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()