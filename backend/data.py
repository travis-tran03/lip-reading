import cv2 as cv
import glob
import os
import matplotlib.image as mpimg




def allFolders():
    type = ["phrases", "words"]
    allImages = []


    for j in range(2):
        for i in range(1,12):

            num = i

            if(i == 11):
                num -= 1

            num = str(num).zfill(2)

            if(i == 3):
                continue

            realNum = str(i).zfill(2)
            path = f"C:/Users/travi/OneDrive/Documents/GitHub/lip-reading/MIRACL-VC1_all_in_one/F{realNum}/{type[j]}/{num}/{num}"
            allImages = load_images(path, allImages)


    return allImages


def load_images(folder, array):

    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            array.append(img)
    return array


testImages = allFolders()
print(len(testImages))
cv.imshow("test", testImages[455])
cv.waitKey(0)


#image = load_images(path)
#cv.imshow("test", image[3])
#cv.waitKey(0)
