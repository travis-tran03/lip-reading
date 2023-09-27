import cv2 as cv
import numpy as np

def cropSingle(img):

    croppedImg = None

    greyImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face = faceCascade.detectMultiScale(greyImg, minSize=(90, 90), maxSize=(90, 90))

    for (x, y, w, h) in face:
        cv.rectangle(greyImg, (x, y), (x+w, y+h), (0, 0, 255), 2)
        face = greyImg[y:y + h, x:x + w]
        #face = np.resize(face, (82, 82))
        #cv.imshow("face", face)
        #cv.waitKey(500)
        croppedImg = face

    return croppedImg

def crop(imgArray):

    croppedArray = []

    for i in range(len(imgArray)):
        greyImg = cv.cvtColor(imgArray[i], cv.COLOR_BGR2GRAY)

        faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face = faceCascade.detectMultiScale(greyImg, minSize=(92, 92))

        for (x, y, w, h) in face:
            cv.rectangle(greyImg, (x, y), (x+w, y+h), (0, 0, 255), 2)
            face = greyImg[y:y + h, x:x + w]
            #face = np.resize(face, (82, 82))
            #cv.imshow("face", face)
            #cv.waitKey(500)
            croppedArray.append(face)

    return croppedArray