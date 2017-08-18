from tools import imutils
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--puzzle", required = True)
ap.add_argument("-w", "--waldo", required = True)
args = vars(ap.parse_args())

puzzle = cv2.imread(args["puzzle"])
waldo = cv2.imread(args["waldo"])

(waldoHeight, waldoWidth) = waldo.shape[:2]
'''
cv2.imshow("puzzle", puzzle)
cv2.imshow("waldo", waldo)
print(waldo.shape)
'''

'''
1.(CV_SQDIFF 和 CV_SQDIFF_NORMED) 最低的数值标识最好的匹配. 
2.(CV_TM_CCORR, CV_TM_CCORR 和 CV_TM_CCOEFF)越大的数值代表越好的匹配.
'''
result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCOEFF)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

'''
print(type(result))
print(str(minVal)+", "+str(maxVal)+", "+str(minLoc)+", "+str(maxLoc))
'''

tl = maxLoc
br = (tl[0]+waldoWidth, tl[1]+waldoHeight)
roi = puzzle[tl[1]:br[1], tl[0]:br[0]]

mask = np.zeros(puzzle.shape, dtype = "uint8")
puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)
puzzle[tl[1]:br[1], tl[0]:br[0]] = roi
cv2.imshow("Puzzle", imutils.resize(puzzle, height = 650))
cv2.imshow("Waldo", waldo)
cv2.waitKey(0)