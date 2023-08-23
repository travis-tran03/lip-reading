import cv2 as cv
import glob
import os
import matplotlib.image as mpimg


def getAllFolders():
    personNum = 1

    type = ['phrases', 'words']

    phrasesImgs = manageImgs([], personNum, type[0])

    wordsImgs = manageImgs([], personNum, type[1])

    allImages = phrasesImgs + wordsImgs

def manageImgs(imgs, personNum, type):
    if (personNum == 12):
        return imgs
    
    if (personNum == 3):
        personNum += 1

    num = 1
    repeatNum = 1

    imgs += getImgs(imgs, personNum, type, num, repeatNum)

    personNum += 1

    manageImgs(imgs, personNum)

def getImgs(imgs, personNum, type, num, repeatNum):
    if (num == 10):
        return imgs
    
    if (repeatNum > 10):
        num += 1
        repeatNum = 1


    path = f"C:/Users/travi/OneDrive/Documents/GitHub/lip-reading/MIRACL-VC1_all_in_one/F{personNum}/{type}/{num}/{repeatNum}"
    imgs += load_images(path, imgs)

    repeatNum += 1

    getImgs(imgs, personNum, num, repeatNum)


class Image:
    def __init__(self, type, id, img):
        self.type = type
        self.id = id
        self.img = img
    
    def getImg(self):
        return self.img

def allFolders():
    type = ["phrases", "words"]
    allImages = []


    for j in range(2):
        for i in range(1,12):
            for k in range(1,11):

                num1 = i
                num2 = k

                if(i == 11):
                    num1 -= 1

                num1 = str(num1).zfill(2)
                num2 = str(num2).zfill(2)

                if(i == 3):
                    continue

                realNum = str(i).zfill(2)
                path = f"C:/Users/travi/OneDrive/Documents/GitHub/lip-reading/MIRACL-VC1_all_in_one/F{realNum}/{type[j]}/{num1}/{num2}"
                loadedImgs = load_images(path, [])
                for image in loadedImgs:
                    test = Image(type[j], num1, image)
                    allImages.append(test)


    return allImages


def load_images(folder, array):

    for filename in os.listdir(folder):

        if ("depth" in filename):
            break

        img = mpimg.imread(os.path.join(folder, filename))

        if img is not None:
            array.append(img)
    return array


testImages = allFolders()
print(len(testImages))

test = cv.cvtColor(testImages[0], cv.COLOR_BGR2GRAY)


faceCascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face = faceCascade.detectMultiScale(test, 1.1, 4)

for (x, y, w, h) in face:
    cv.rectangle(testImages[0], (x, y), (x+w, y+h), (0, 0, 255), 2)
    face = testImages[0][y:y + h, x:x + w]
    cv.imshow("face",face)
    #cv.imwrite('face.jpg', face)

cv.imshow("test", testImages[1000])
cv.waitKey(0)


#image = load_images(path)
#cv.imshow("test", image[3])
#cv.waitKey(0)
