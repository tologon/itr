# Author:   Tologon Eshimkanov (https://github.com/tologon)
# Course:   COMP 3770-01 - Introduction to Artificial Intelligence
# School:   Wentworth Institute of Technology
# Project:  Image Text Recognition

# required package(s)
import sys, cv2
import numpy as np

# ignore sys.argv[0] as it is a name of an invoked Python script
# print 'Number of arguments:', len(sys.argv[1:]), 'arguments.'
# print 'Argument list:', str(sys.argv[1:])

# constants
DEFAULT_SINGLE_DIGIT = 'default_single_digit.png'
RESULT_TYPE = 'curve'
DEFAULT_COLOR = (0, 255, 0) # RGB values; doesn't matter on grayscale

class Pipeline:
    """
    The pipeline takes an image as input and detects
    any present digits with some degree of accuracy.
    """

    def __init__(self, image = DEFAULT_SINGLE_DIGIT):
        """ Reads and stores an image for future processing. """
        # possible options for image reading:
        # cv2.IMREAD_COLOR      : Loads a color image (default)
        # cv2.IMREAD_GRAYSCALE  : Loads image in grayscale mode
        # cv2.IMREAD_UNCHANGED  : Loads image as such including alpha channel

        # add check for image existence on a given path
        self.image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    def detectRegions(self):
        """
        Detects MSER regions and subsequently converting
        those regions into convex hulls, rectangles, and contours.
        """
        mser = cv2.MSER_create()
        bboxes = None # no documentation available
        self.regions = mser.detectRegions(self.image, bboxes)
        self.hulls = self.regionsToHulls()
        self.rectangles = self.regionsToRectangles()
        self.contours = self.regionsToContours()

    def regionsToHulls(self):
        """ Converts present MSER regions into convex hulls. """
        return [ cv2.convexHull( r.reshape(-1, 1, 2) ) for r in self.regions ]

    def regionsToRectangles(self):
        """ Converts present MSER regions into rectangles. """
        return [ cv2.boundingRect(region) for region in self.regions ]

    def regionsToContours(self):
        """ Converts present MSER regions into contours. """
        closed = True
        contours = []
        for region in self.regions:
            epsilon = 0.01 * cv2.arcLength(region, True)
            contours.append( cv2.approxPolyDP(region, epsilon, closed) )
        return contours

    # TODO: refactor resultType into something more sensible (w/o user input)
    def drawResult(self, result, resultType = RESULT_TYPE):
        """ Draws a given result on the original image. """
        imageCopy = self.image.copy()
        isClosed = 1 # no documentation available
        if resultType == 'curve':
            cv2.polylines(imageCopy, result, isClosed, DEFAULT_COLOR)
        elif resultType == 'straight':
            self.drawRectangles(imageCopy, result)
        cv2.imshow('Result', imageCopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def drawRectangles(self, image, rectangles):
        """ Draws rectangles on the image. """
        for rectangle in rectangles:
            x, y, w, h = rectangle
            topLeftCorner = (x, y)
            bottomRightCorner = (x + w, y + h)
            cv2.rectangle(image, topLeftCorner, bottomRightCorner, DEFAULT_COLOR)


if __name__ == "__main__":
    image, pipeline = None, None
    if len(sys.argv) > 1:
        image = str(sys.argv[1])
        pipeline = Pipeline(image)
    else:
        pipeline = Pipeline()
    pipeline.detectRegions()

    print "regions: {}".format( len(pipeline.regions) )
    print "hulls: {}".format( len(pipeline.hulls) )
    print "rectangles: {}".format( len(pipeline.rectangles) )
    print "contours: {}".format( len(pipeline.contours) )

    pipeline.drawResult( pipeline.hulls, 'curve' )
    # pipeline.drawResult( pipeline.contours, 'curve' )
    pipeline.drawResult( pipeline.rectangles, 'straight' )
