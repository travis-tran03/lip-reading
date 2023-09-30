from resizing import cropSingle
import cv2 as cv

path = "C:/Users/travi/OneDrive/Documents/GitHub/lip-reading/backend/allData/color_005.jpg"

singleImage = cv.imread(path)

singleImage = cropSingle(singleImage, (91,91))

cv.imshow("fSingleCrop", singleImage)
cv.waitKey(0)

cv.imwrite("singleImage.jpg", singleImage)

