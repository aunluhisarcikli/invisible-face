
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')


cap = cv2.VideoCapture(0)


count = 0
background = 0

for i in range(60):
    return_val, background = cap.read()
    if return_val == False:
        continue

time.sleep(1.0) # giving some time for camera


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Define the codec and create VideoWriter object. The output is stored in 'out.avi' file.
out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 21, (frame_width, frame_height))


while True:

    ret, frame = cap.read()
    #frame = np.flip(frame,axis=0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0) # convert to grayscale

    for rect in rects:
        # get landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)


        lip = shape[48:60] # lip landmarks values
        eye_left = shape[36:42] #left eye landmarks values
        eye_right = shape[42:48] #right eye landmarks values

        cv2.drawContours(frame, [lip], -1, (255, 40, 50), -1) # draw the lip
        cv2.drawContours(frame, [eye_left], -1, (255, 40, 50), -1) # draw the left eye
        cv2.drawContours(frame, [eye_right], -1, (255, 40, 50), -1) # draw the right eye


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert to HSV


    # ranges should be carefully chosen
    # setting the lower and upper range for mask1
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)


    # Refining the mask corresponding to the detected the color
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),
                                                            np.uint8), iterations=1)
    mask1 = cv2.dilate(mask1, np.ones((5, 5), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)

    # Generating the final output
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)
    final = cv2.addWeighted(res1, 1, res2, 1, 0)

    # show the outputs
    cv2.imshow('invisible', final)
    cv2.imshow("Frame", frame)

    out.write(final) # write final to out
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()