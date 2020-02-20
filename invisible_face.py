import cv2
import numpy as np

#load model
face_cascade = cv2.CascadeClassifier('model\haarcascade_frontalface_alt.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)


background = 0

# capturing the background in range of 60
# you should have video that have some seconds
# dedicated to background frame so that it
# could easily save the background image
for i in range(60):
    return_val, background = cap.read()
    if return_val == False:
        continue

#background = np.flip(background, axis=1)  # flipping of the frame


# loop runs if capturing has been initialized.
while 1:
    # reads frames from a camera
    ret, img = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # To draw a filled rectangle in a face
        # should be choose color in mask range
      #  cv2.rectangle(img, (x , y ), (x + w , y + h ), (70, 255, 255), -1)
        cv2.circle(img, (int((2*x +w)/2),int((2*y +h)/2)), int((x +w)/4), (255, 40, 50), -1)


    #convert to HSV format
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ranges should be carefully chosen
    # setting the lower and upper range for mask1
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    '''
    #other colors can also be used
    low_red = np.array([100, 40, 40])
    high_red = np.array([100, 255, 255])
    
    lower_yellow = np.array([15, 42, 30])
    upper_yellow = np.array([30, 255, 255])
    mask1 = cv2.inRange(hsv, lower_color, upper_color)
    
    # a second interval can also be determined.
    low_red2 = np.array([160, 45, 40])
    high_red2 = np.array([180, 255, 255])
    
    lower_yellow2 = np.array([55, 52, 72])
    upper_yellow2 = np.array([102, 255, 255])
    mask2 = cv2.inRange(hsv, lower_color, upper_color)
    
    # mask1 = mask1 + mask2
    '''

    # Refining the mask corresponding to the detected the color
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3),
                                                            np.uint8), iterations=2)
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)

    # Generating the final output
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final = cv2.addWeighted(res1, 1, res2, 1, 0)


    cv2.imshow('invisible', final)
    cv2.imshow('img', img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()