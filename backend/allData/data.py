import cv2 as cv
import glob
import os
import matplotlib.image as mpimg
import numpy as np
from dotenv import load_dotenv

from resizing import crop


phrases = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye", "i love this game", "nice to meet you", "you are welcome", "how are you", "have a good time"]
words = ["begin", "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "wed"]

load_dotenv('backend/.env')

basePath = os.getenv("FOLDERPATH")

class Frame:
    def __init__(self, type, id, expression, img):
        self.type = type
        self.id = id
        self.expression = expression
        self.img = img
    
    def getImg(self):
        return self.img
    
def test(arr, dir):
    x = os.scandir(dir)

    for file in x:
        if file.is_dir():
            test(arr, file.path)
        if file.is_file() and '.jpg' in file.name:
            arr.append(cv.imread(file.path))

    return arr


def getAllFolders():
    personNum = 1

    type = ['phrases', 'words']
    
    phrasesImgs = manageImgs([], personNum, type[0])

    wordsImgs = manageImgs([], personNum, type[1])

    allImages = phrasesImgs + wordsImgs

    return allImages

def manageImgs(imgs, personNum, type):
    if (personNum == 12):
        return imgs
    
    if (personNum == 3):
        personNum += 1

    num = 1
    repeatNum = 1

    

    imgs = getImgs(imgs, personNum, type, num, repeatNum)

    personNum += 1

    imgs = manageImgs(imgs, personNum, type)

    return imgs

def getImgs(imgs, personNum, type, num, repeatNum):
    if (num == 10):
        return imgs
    
    if (repeatNum > 10):
        num += 1
        repeatNum = 1


    path = os.path.join(basePath, f"F{str(personNum).zfill(2)}/{type}/{str(num).zfill(2)}/{str(repeatNum).zfill(2)}")
    imgs = load_images(path, imgs)

    repeatNum += 1

    imgs = getImgs(imgs, personNum, type, num, repeatNum)

    return imgs

def allFolders():
    type = ["phrases", "words"]
    allImages = []

    print(f"Path .env: ", basePath)

    for j in range(2):
        for i in range(1,12):
            for k in range(1,11):
                for m in range(1, 11):
                    phraseIndex = 0

                    num1 = str(k).zfill(2)
                    num2 = str(m).zfill(2)

                    if(i == 3):
                        continue

                    realNum = str(i).zfill(2)
                    path = os.path.join(basePath, f"F{realNum}/{type[j]}/{num1}/{num2}")
                    loadedImgs = load_images(path, [])
                    for image in loadedImgs:
                        test = Frame(type[j], i, phrases[phraseIndex], image)
                        allImages.append(test)
                    phraseIndex += 1


    return allImages


def load_images(folder, array):
    for filename in os.listdir(folder):

        if ("depth" in filename):
            break

        img = mpimg.imread(os.path.join(folder, filename))

        if img is not None:
            array.append(img)
    return array

def loadData(label, labelString):
    count = 0
    result = np.empty([2, 1, 1]) # 2: videos/labels, 1: repeat, 1: frames 0-10

    for index in range(0, len(label)):
        expression = label[index]
        #expLabel = [ i for i, j in locals().items() if j == label][0]
        for person in range(1, 12):
            if person == 3:
                continue
            for expressionNum in range(1, 11):
                for repeat in range(1, 11):
                    # Increase Shape

                    path = os.path.join(basePath, f"F{str(person).zfill(2)}/{labelString}/{str(expressionNum).zfill(2)}/{str(repeat).zfill(2)}")
                    tempArr = load_images(path, [])

                    result.reshape((-1, -1, (-1 + 1)))
                    result[0][count] = tempArr
                    result[1][count] = labelString

                    count += 1


testImages = getAllFolders()

#croppedArr = crop(testImages)

singleImage = [mpimg.imread('C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/MIRACL-VC1_all_in_one/F01/phrases/03/01/color_001.jpg')]
print(singleImage)
croppedImage = crop(singleImage)
croppedImage = np.array(croppedImage)
#cv.imshow('TEST', testImages[10000])
#cv.imshow('cropped', testImages[10000])
#print(croppedImage)
cv.imshow('test', croppedImage[0])
cv.waitKey(0)

#phrasesArr = loadData(phrases, 'phrases')
#wordsArr = loadData(words, 'words')

#print(phrasesArr.shape)
#print(wordsArr.shape)


