import cv2, math, mnist
import numpy as np
from sklearn import svm

img = cv2.imread('test_image8.png', 0)
# Pipeline - MSER detection
mser = cv2.MSER_create()

regions = mser.detectRegions(img, None)
hulls = [ cv2.convexHull( p.reshape(-1, 1, 2) ) for p in regions ]

vis = img.copy()
cv2.polylines(vis, hulls, 1, (0, 255, 0))
cv2.imshow('img', vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ------------------------------------------------------------------------------
# Pipeline - non-text regions removed based on geometric properties
# filter by aspect ratio
def aspectRatio(convexHull):
    x, y, w, h = cv2.boundingRect(convexHull)
    # if float(w)/h > aspectRatioThreshold:
    #     print "aspect ratio: {}".format( float(w) / h )
    return float(w) / h

aspectRatioThreshold = 0.29

print "BEFORE filter w/ aspect ratio: {}".format( len(hulls) )
hulls = [hull for hull in hulls if aspectRatio(hull) > aspectRatioThreshold]
print "AFTER filter w/ aspect ratio: {}".format( len(hulls) )

# vis = img.copy()
# cv2.polylines(vis, hulls, 1, (0, 255, 0))
# cv2.imshow('img', vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# filter by extent
def extent(convexHull):
    area = cv2.contourArea(convexHull)
    x, y, w, h = cv2.boundingRect(convexHull)
    rectArea = w * h
    # print "extent: {}".format( float(area) / rectArea )
    return float(area) / rectArea

lowExtentThreshold = 0.3
highExtentThreshold = 0.59

print "BEFORE filter w/ extent: {}".format( len(hulls) )
hulls = [hull for hull in hulls if extent(hull) < lowExtentThreshold or extent(hull) > highExtentThreshold]
print "AFTER filter w/ extent: {}".format( len(hulls) )

# vis = img.copy()
# cv2.polylines(vis, hulls, 1, (0, 255, 0))
# cv2.imshow('img', vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# filter by solidity
def solidity(convexHull):
    area = cv2.contourArea(convexHull)
    hull = cv2.convexHull(convexHull)
    hullArea = cv2.contourArea(hull)
    # print "solidity: {}".format( float(area) / hullArea )
    return float(area) / hullArea

solidityThreshold = 1.1

print "BEFORE filter w/ solidity: {}".format( len(hulls) )
hulls = [hull for hull in hulls if solidity(hull) < solidityThreshold]
print "AFTER filter w/ solidity: {}".format( len(hulls) )

# vis = img.copy()
# cv2.polylines(vis, hulls, 1, (0, 255, 0))
# cv2.imshow('img2', vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ------------------------------------------------------------------------------
# Pipeline - non-text regions removed based on stroke width variation
def strokeWidthVariation(convexHull):
    convexHull = convexHull.transpose(2,0,1).reshape(2,-1)
    convexHull = np.uint8(convexHull)
    edge = cv2.Canny(convexHull, 100, 200)
    return edge

edges = [strokeWidthVariation(hull) for hull in hulls]

def strokeWidthMetric(stroke):
    # print "metric: {}".format( np.std(stroke) / np.mean(stroke) )
    return np.std(stroke) / np.mean(stroke)

strokeWidthThreshold = 0.99

print "BEFORE filter w/ stroke width variation: {}".format( len(hulls) )
hulls = [hull for hull, edge in zip(hulls, edges) if strokeWidthMetric(edge) > strokeWidthThreshold or math.isnan(strokeWidthMetric(edge)) ]
print "AFTER filter w/ stroke width variation: {}".format( len(hulls) )

vis = img.copy()
cv2.polylines(vis, hulls, 1, (0, 255, 0))
cv2.imshow('img3', vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ------------------------------------------------------------------------------
