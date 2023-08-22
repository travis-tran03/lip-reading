import cv2 as cv
import glob
import os
import matplotlib.image as mpimg




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
                allImages = load_images(path, allImages)


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
cv.imshow("test", testImages[1000])
cv.waitKey(0)


#image = load_images(path)
#cv.imshow("test", image[3])
#cv.waitKey(0)
