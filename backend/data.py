import cv2 as cv
import glob
import os
import matplotlib.image as mpimg


phrases = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye", "i love this game", "nice to meet you", "you are welcome", "how are you", "have a good time"]
words = ["begin", "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "wed"]

class Frame:
    def __init__(self, type, id, expression, img):
        self.type = type
        self.id = id
        self.expression = expression
        self.img = img
    
    def getImg(self):
        return self.img

def allFolders():
    type = ["phrases", "words"]
    allImages = []


    for j in range(2):
        for i in range(1,12):
            for k in range(1,11):

                phraseIndex = 0
                num1 = i
                num2 = k

                if(i == 11):
                    num1 -= 1

                num1 = str(num1).zfill(2)
                num2 = str(num2).zfill(2)

                if(i == 3):
                    continue

                realNum = str(i).zfill(2)
                path = f"C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/backend/MIRACL-VC1_all_in_one/F{realNum}/{type[j]}/{num1}/{num2}"
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


#testImages = getAllFolders()
#print(len(testImages))

#print(testImages[500].id)
#cv.imshow("img3", testImages[3])
#cv.imshow("img20000", testImages[20000])
#cv.waitKey(0)

'''
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
#cv.waitKey(0)'''
