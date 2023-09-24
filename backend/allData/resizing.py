import cv2 as cv
import numpy as np

def crop(imgArray):

    croppedArray = []

    for image in imgArray:
        greyImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        face = faceCascade.detectMultiScale(greyImg, minSize=(80, 80), maxSize=(80, 80))

        for (x, y, w, h) in face:
            cv.rectangle(greyImg, (x, y), (x+w, y+h), (0, 0, 255), 2)
            face = greyImg[y:y + h, x:x + w]
            print(face.shape)
            #face = np.resize(face, (82, 82))
            #cv.imshow("face", face)
            #cv.waitKey(0)
            croppedArray.append(face)

    return croppedArray