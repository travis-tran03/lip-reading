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

def load_images(folder, array, notCropped):
    for filename in os.listdir(folder):

        if ("depth" in filename):
            break

        img = mpimg.imread(os.path.join(folder, filename))

        greyImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        cropImg = cropSingle(greyImg, (90, 90))
        if cropImg is not None:
            (width, height) = np.array(cropImg).shape
            left = int(width/2)
            right = int(height/2)
            left2 = int(left/2)
            left3 = int((left+(left/2)))
            cropImg = cropImg[right:height, left2:left3]
            cropImg = cv.resize(cropImg, (91, 91))

            array.append(cropImg)
        else:
            cropImg = cropSingle(greyImg, (100, 100))
            
            
            if (len(notCropped) > 0):
            
                notCropped.append({'img': img, 'greyImg': greyImg, 'cropImg': cropImg, 'index': (notCropped[-1]['index'] + 1)})
            else:
                notCropped.append({'img': img, 'greyImg': greyImg, 'cropImg': cropImg, 'index': len(array)})

    return array, notCropped



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

