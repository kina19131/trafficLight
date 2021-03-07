import cv2
import imutils
import numpy as np

#LETTING THE COMPUTER KNOW THIS IS THE CAMERA I AM USING
cap = cv2.VideoCapture(0)

## WHILE CAMERA IS RUNNING ##
while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ## DEFINING COLOR RANGE ##
    #Red Color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])

    #Green Color
    low_green = np.array([25, 65, 72])
    high_green = np.array([102, 255, 255])

    #Yellow Color
    low_yellow = np.array([20, 110, 100])
    high_yellow = np.array([30, 255, 255])

    ## EXTRACTING THE COLOR OUT ##
    #Mask
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)

    ## CONTOUR ##
    cntsR = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntsR = imutils.grab_contours(cntsR)

    cntsG = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntsG = imutils.grab_contours(cntsG)

    cntsY = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntsY = imutils.grab_contours(cntsY)

    for c in cntsR:
        areaR = cv2.contourArea(c)
        if areaR > 3000:
            cv2.drawContours(frame, [c], -1, (255,255,255), 3)
            M = cv2.moments(c)

            ## FIND CENTER ##
            cx = int(M["m10"]/ M["m00"])
            cy = int(M["m01"]/M["m00"])

            ## WRITE TEXT ##
            cv2.putText(frame, "RED", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    for c in cntsG:
        areaG = cv2.contourArea(c)
        if areaG > 3000:
            cv2.drawContours(frame, [c], -1, (255,255,255), 3)
            M = cv2.moments(c)

            ## FIND CENTER ##
            cx = int(M["m10"]/ M["m00"])
            cy = int(M["m01"]/M["m00"])

            ## WRITE TEXT ##
            cv2.putText(frame, "GREEN", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)


    for c in cntsY:
        areaY = cv2.contourArea(c)
        if areaY > 3000:
            cv2.drawContours(frame, [c], -1, (255,255,255), 3)
            M = cv2.moments(c)

            ## FIND CENTER ##
            cx = int(M["m10"]/ M["m00"])
            cy = int(M["m01"]/M["m00"])

            ## WRITE TEXT ##
            cv2.putText(frame, "Y", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    ## SHOW ON THE SCREEN ##
    cv2.imshow("result", frame)

    #IF YOU PRESS "ESC" IT EXITS
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()