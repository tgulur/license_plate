# This program aims to find the license plate in a given image and return the text of the license plate using the OpenCV library and the Tesseract OCR library.

import cv2 as cv
import numpy as np
import pytesseract
import imutils

# Read image and convert to greyscale
img = cv.imread('plate12.jpg')
img = cv.resize(img, (620,480))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Use bilateralFilter to remove noise
gray = cv.bilateralFilter(gray, 11, 17, 17)


# Apply edge detection using Canny edge detector
edged = cv.Canny(gray, 30, 200)

# Find contours in the edged image, keep only the largest ones, and initialize the screen contour
cnts, _ = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

screenCnt = 0

# Loop over the countours and find the one that is the license plate
for c in cnts:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.03 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print("No license plate detected")
else:
    detected = 1
    cv.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
# Mask the image using the screen contour
mask = np.zeros(gray.shape, np.uint8)
newImage = cv.drawContours(mask, [screenCnt], 0, 255, -1)
newImage = cv.bitwise_and(img, img, mask = mask)
# Crop the image to only contain the license plate
(x,y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray[topx:bottomx+1, topy:bottomy+1]



# Show image
cv.imshow('Test', cropped)
cv.waitKey(0)
