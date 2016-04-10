# ------------------------------------------------------------------------------
# Author:   Tologon Eshimkanov (https://github.com/tologon)
# Course:   COMP 3770-01 - Introduction to Artificial Intelligence
# School:   Wentworth Institute of Technology
# Project:  Simplified Digit Recognition
# ------------------------------------------------------------------------------

# required package(s)
import cv2
import numpy as np

def aspectRatio(convexHull):
    """
    Aspect ratio is the ratio of width to height.
        width / height = aspect ratio
    A convex hull is not a shape from which values can be easily extracted.
    Therefore, a convex hull is transformed into a bounding rectangle,
    from which its width and height are derived.
    """
    x, y, w, h = cv2.boundingRect(convexHull)
    return float(w) / h

def extent(convexHull):
    """
    Extent is the ratio of contour area to bounding rectangle area.
    """
    area = cv2.contourArea(convexHull)
    x, y, w, h = cv2.boundingRect(convexHull)
    rectArea = w * h
    return float(area) / rectArea

def solidity(convexHull):
    """
    Solidity is the ratio of contour area to its convex hull area.
    """
    area = cv2.contourArea(convexHull)
    hullArea = cv2.contourArea(convexHull)
    return float(area) / hullArea

def strokeWidthVariation(convexHull):
    """
    Finds edges in a convex hull by converting
    convex hull into an approriate shape.
    """
    convexHull = convexHull.transpose(2,0,1).reshape(2,-1)
    convexHull = np.uint8(convexHull)
    edge = cv2.Canny(convexHull, 100, 200)
    return edge

def strokeWidthMetric(stroke):
    """ Calculates stroke width variation for a given stroke. """
    return np.std(stroke) / np.mean(stroke)
