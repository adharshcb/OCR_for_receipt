import cv2
import numpy as np
import pytesseract
import os

per = 25

img_q = cv2.imread('images/temp2.jpg')
# w, h, c = img_q.shape
img_q = cv2.resize(img_q,(1280,1033))
# print(img_q)

# cv2.imshow('output',img_q)

orb = cv2.ORB_create(1000)

keyP1 , DesP1 = orb.detectAndCompute(img_q, None)

# imgP1 = cv2.drawKeypoints(img_q, keyP1, None)
# cv2.imshow('1', imgP1)

path = 'images'
pic_list = os.listdir(path=path)


for j, y in enumerate(pic_list):
    img = cv2.imread(path+'/'+y)
    # print(y,img.shape)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow(y,img)
    keyP2, DesP2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    dfMatch = bf.match(DesP2, DesP1)
    dfMatch = sorted(dfMatch, key = lambda x: x.distance)
    good = dfMatch[:int(len(dfMatch)*(per/100))]
    img_match = cv2.drawMatches(img, keyP2, img_q, keyP1, good[:100], None, flags=2)
    # cv2.imshow(y, img_match)

    w, h, c = img.shape

    srcPoints = np.float32([keyP2[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dstPoints = np.float32([keyP1[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
    
    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

    img_scan = cv2.warpPerspective(img, M, (w, h))
    img_scan = cv2.resize(img_scan,(w//3, h//3))
    cv2.imshow(y, img_scan)




# print(pic_list)

cv2.waitKey(0)
cv2.destroyAllWindows()