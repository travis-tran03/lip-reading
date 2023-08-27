import cv2 as cv
import numpy as np

def crop(imgArray):

    croppedArray = []

    for image in imgArray:
        greyImg = cv.cvtColor(image.img, cv.COLOR_BGR2GRAY)

        faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        face = faceCascade.detectMultiScale(greyImg, 1.1, 4)

        for (x, y, w, h) in face:
            cv.rectangle(image.img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            face = image.img[y:y + h, x:x + w]
            croppedArray.append(face)
            #cv.imshow("face",face)
            #cv.waitKey(0)

    return croppedArray


#cv.imshow("test", testImages[0].img)
#cv.waitKey(0)