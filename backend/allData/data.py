import cv2 as cv
import os
import matplotlib.image as mpimg
import numpy as np
from dotenv import load_dotenv

from .resizing import crop, cropSingle


phrases = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye", "i love this game", "nice to meet you", "you are welcome", "how are you", "have a good time"]
words = ["begin", "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "wed"]

load_dotenv('backend/.env')

basePath = os.getenv("FOLDERPATH")

def lipCrop(img, width, height, x=0, y=0):
    left = int(width/2)
    right = int(height/2)
    lipCrop = img[right:height + y, int(left/2):int((left+(left/2))) + x]
    lipCrop = cv.resize(lipCrop, (91, 91))

    return lipCrop

def load_images(folder, array, notCropped):
    lipArray = []
    upLipArr = []
    leftLipArr = []
    leftShiftArr = []
    upShiftArr = []

    for filename in os.listdir(folder):

        if ("depth" in filename):
            break

        img = mpimg.imread(os.path.join(folder, filename))

        cropImg = cropSingle(img, (90, 90))

        if cropImg is not None:
            
            (width, height, channels) = np.array(cropImg).shape

            TLeft = np.float32([[1, 0, 0], [0, 1, -10]])
            TUp = np.float32([[1, 0, 10], [0, 1, 0]])

            leftShift = cv.warpAffine(cropImg, TLeft, (width, height))
            upShift = cv.warpAffine(cropImg, TUp, (width, height))

            lipImgCropped = lipCrop(cropImg, width, height)
            leftLipImgCropped = lipCrop(leftShift, width, height, -10, 0)
            upLipImgCropped = lipCrop(upShift, width, height, 0, 10)

            leftShiftArr.append(leftShift)
            leftLipArr.append(leftLipImgCropped)
            upShiftArr.append(upShift)
            upLipArr.append(upLipImgCropped)
            lipArray.append(lipImgCropped)
            array.append(cropImg)
        else:
            cropImg = cropSingle(img, (100, 100))
            
            
            if (len(notCropped) > 0):
            
                notCropped.append({'img': img, 'greyImg': img, 'cropImg': cropImg, 'index': (notCropped[-1]['index'] + 1)})
            else:
                notCropped.append({'img': img, 'greyImg': img, 'cropImg': cropImg, 'index': len(array)})

    return array, lipArray, leftShiftArr, leftLipArr, upShiftArr, upLipArr, notCropped



def getAllFolders():
    personNum = 1

    type = ['phrases', 'words']
    
    phrasesImgs, phrasesLabels = manageImgs([], [], personNum, type[0])

    wordsImgs, wordsLabels = manageImgs([], [], personNum, type[1])

    allImages = phrasesImgs + wordsImgs
    allLabels = phrasesLabels + wordsLabels

    return allImages, allLabels

def manageImgs(imgs, labels, personNum, type):
    if (personNum == 12):
        return imgs, labels
    
    if (personNum == 3):
        personNum += 1

    num = 1
    repeatNum = 1

    

    imgs, labels = getImgs(imgs, labels, personNum, type, num, repeatNum)

    personNum += 1

    imgs, labels = manageImgs(imgs, labels, personNum, type)

    return imgs, labels

def getImgs(imgs, labels, personNum, type, num, repeatNum):
    if (num == 10):
        return imgs, labels
    
    if (repeatNum > 10):
        num += 1
        repeatNum = 1


    path = os.path.join(basePath, f"F{str(personNum).zfill(2)}/{type}/{str(num).zfill(2)}/{str(repeatNum).zfill(2)}")
    newImgs = load_images(path, [])

    for i in range(len(newImgs)):
        imgs.append(newImgs[i])
        if (type == 'phrases'):
            labels.append(phrases[num-1])
        if (type == 'words'):
            labels.append(words[num-1])

    repeatNum += 1

    imgs, labels = getImgs(imgs, labels, personNum, type, num, repeatNum)

    return imgs, labels

