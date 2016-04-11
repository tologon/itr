# ------------------------------------------------------------------------------
# Author:       Tologon Eshimkanov (https://github.com/tologon)
# Course:       COMP 3770-01 - Introduction to Artificial Intelligence
# School:       Wentworth Institute of Technology
# Project:      Simplified Digit Recognition
# Description:
#   This file contains useful utility functions, mostly applied
#   to filter out convex hulls either by geometric or text-based properties.
# ------------------------------------------------------------------------------

# required package(s)
import cv2
import numpy as np

def aspect_ratio(convex_hull):
    """
    Aspect ratio is the ratio of width to height.
        width / height = aspect ratio
    A convex hull is not a shape from which values can be easily extracted.
    Therefore, a convex hull is transformed into a bounding rectangle,
    from which its width and height are derived.
    """
    x, y, w, h = cv2.boundingRect(convex_hull)
    return float(w) / h

def extent(convex_hull):
    """
    Extent is the ratio of contour area to bounding rectangle area.
    """
    area = cv2.contourArea(convex_hull)
    x, y, w, h = cv2.boundingRect(convex_hull)
    rect_area = w * h
    return float(area) / rect_area

def solidity(convex_hull):
    """
    Solidity is the ratio of contour area to its convex hull area.
    """
    area = cv2.contourArea(convex_hull)
    hull_area = cv2.contourArea(convex_hull)
    return float(area) / hull_area

def stroke_width_variation(convex_hull):
    """
    Finds edges in a convex hull by converting
    convex hull into an approriate shape.
    """
    convex_hull = convex_hull.transpose(2,0,1).reshape(2,-1)
    convex_hull = np.uint8(convex_hull)
    edges = cv2.Canny(convex_hull, 100, 200, L2gradient=True)
    return edges

def SWV_metric(stroke):
    """ Calculates stroke width variation for a given stroke. """
    return np.std(stroke) / np.mean(stroke)
