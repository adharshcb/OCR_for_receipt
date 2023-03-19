import cv2
import numpy as np
import pytesseract
import os

per = 25
roi = [[[(720, 800), (1150, 915)], 'text', 'date'],
       [[(720, 915), (1150, 1030)], 'text', 'time'],
       [[(720, 1030), (1150, 1140)], 'text', 'bill'],
       [[(720, 1240), (1150, 1360)], 'text', 'location'],
       [[(720, 1360), (1390, 1475)], 'text', 'l_description'],
       [[(720, 1475), (1390, 1580)], 'text', 'item_code'],
       [[(720, 1580), (1650, 1700)], 'text', 'description'],
       [[(720, 1700), (1150, 1800)], 'text', 'quantity']
       ]

img_q = cv2.imread('images/temp2.jpg')
# img_q = cv2.resize(img_q,(550,720))
w, h, c = img_q.shape
# print(img_q)

# cv2.imshow('output',img_q)

orb = cv2.ORB_create(5000)

kp1 , des1 = orb.detectAndCompute(img_q, None)

# imgP1 = cv2.drawKeypoints(img_q, keyP1, None)
# cv2.imshow('1', imgP1)

path = 'images'
pic_list = os.listdir(path=path)


for j, y in enumerate(pic_list):
    img = cv2.imread(path+'/'+y)
    # print(y,img.shape)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow(y,img)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches = sorted(matches, key = lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    img_match = cv2.drawMatches(img, kp2, img_q, kp1, good, None, flags=2)
    # img_match = cv2.resize(img_match,(w//3, h//3))
    # cv2.imshow(y, img_match)

    # w, h, c = img.shape

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

    
    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 2.0)

    # print(y,M)

    img_scan = cv2.warpPerspective(img, M, (w, h))
    # img_scan = cv2.resize(img_scan,(w//3, h//3))
    # cv2.imshow(y, img_scan)

    imgShow = img_scan.copy()
    imgMask = np.zeros_like(imgShow)


    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, r[0][0], r[0][1], (0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        # print(r[0][0][1],r[0][1][0],r[0][0][0],r[0][1][0])
        # [[(720, 800), (1150, 915)], 'text', 'date']
        imgCrop = imgShow[r[0][0][1]:r[0][1][1],r[0][0][0]:r[0][1][0]]

        cv2.imshow(str(x)+y,imgCrop)
        cv2.waitKey(0)
    
    # imgShow = cv2.resize(imgShow,(w//3, h//3))
    # cv2.imshow(y,imgShow)


# print(pic_list)

cv2.waitKey(0)
cv2.destroyAllWindows()