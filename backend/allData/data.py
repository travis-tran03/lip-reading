import cv2 as cv
import glob
import os
import matplotlib.image as mpimg
import numpy as np
from resizing import crop
from dotenv import load_dotenv
from timeit import timeit


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



testImages = np.array(allFolders())
testImages2 = np.array(getAllFolders())
print(testImages.shape)
print(len(testImages))
print(testImages2.shape)
print(len(testImages2))

for_loop_time: float = timeit(stmt='allFolders()', globals=globals(), number=3)
recursion_time: float = timeit(stmt='getAllFolders()', globals=globals(), number=3)

print(f'For-Loop: {for_loop_time}')
print(f'Recursion: {recursion_time}')

'''
croppedArray = crop(testImages)
print(len(croppedArray))
croppedArray = np.array(croppedArray, dtype=object)
print(f'CroppedArray Length: {len(croppedArray)}')
print(f'Image 1: {croppedArray[1].shape}')

# cv.namedWindow("Image 1", cv.WINDOW_NORMAL)
# cv.resizeWindow("Image 1", 150, 150)

cv.imshow('Image 1', croppedArray[1])
cv.imshow('Image 61', croppedArray[61])
cv.waitKey(0)

#print(len(testImages))

#print(testImages[500].id)
#cv.imshow("img3", testImages[3])
#cv.imshow("img20000", testImages[20000])
#cv.waitKey(0)
'''
