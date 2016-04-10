# ------------------------------------------------------------------------------
# Author:   Tologon Eshimkanov (https://github.com/tologon)
# Course:   COMP 3770-01 - Introduction to Artificial Intelligence
# School:   Wentworth Institute of Technology
# Project:  Simplified Digit Recognition
# ------------------------------------------------------------------------------

# required package(s)
import cv2, math, classifier
import numpy as np

from utilities import *
from classifier import plt
# ------------------------

# constants
DEFAULT_IMAGE = 'default_single_digit.png'
DEFAULT_COLOR = (0, 255, 0) # RGB values; doesn't matter on grayscale
ASPECT_RATIO_THRESHOLD = 1
LOW_EXTENT_THRESHOLD = 0.3
HIGH_EXTENT_THRESHOLD = 0.59
SOLIDITY_THRESHOLD = 1.1
STROKE_WIDTH_THRESHOLD = 0.99
GROUP_THRESHOLD = 1 # minimum possible number of rectangles - 1

class Pipeline:
    """
    The pipeline takes an image as input and detects
    any present digits with some degree of accuracy.
    """

    def __init__(self, image = DEFAULT_IMAGE):
        """
        Reads and stores an image for future processing.
        possible options for image reading:
            cv2.IMREAD_COLOR      : Loads a color image (default)
            cv2.IMREAD_GRAYSCALE  : Loads image in grayscale mode
            cv2.IMREAD_UNCHANGED  : Loads image as such including alpha channel
        """
        self.image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise Exception( "Cannot find the image {}.".format(image) )

    def detect_regions(self):
        """
        Detects MSER regions and subsequently converting
        those regions into convex hulls, and rectangles.
        """
        mser = cv2.MSER_create()
        bboxes = None # no documentation available
        self.regions = mser.detectRegions(self.image, bboxes)
        self.hulls = self.regions_to_hulls() # hulls == convex hulls
        self.rectangles = self.regions_to_rectangles()

    def regions_to_hulls(self):
        """ Converts present MSER regions into convex hulls. """
        return [ cv2.convexHull( region.reshape(-1, 1, 2) )
                    for region in self.regions ]

    def regions_to_rectangles(self):
        """ Converts present MSER regions into rectangles. """
        return [ cv2.boundingRect(region) for region in self.regions ]

    def hulls_to_rectangles(self):
        """ Converts present convex hull into rectangles. """
        return [ cv2.boundingRect(hull) for hull in self.hulls ]

    def draw_results(self, results):
        """
        Draws given results on the original image.
            if there are no results, throw an exception.
            if results in rectangular form, draw rectangles.
            if results in convex hulls form, draw polylines (i.e. curved lines).
        """
        imageCopy = self.image.copy()
        isClosed = 1 # no documentation available
        if len(results) == 0:
            raise Exception("Cannot draw results due to result's absence.")
        elif len(results[0]) == 4:
            self.draw_rectangles(imageCopy, results)
        else:
            cv2.polylines(imageCopy, results, isClosed, DEFAULT_COLOR)
        cv2.imshow('Results', imageCopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_rectangles(self, image, rectangles):
        """ Draws rectangles on the image. """
        for rectangle in rectangles:
            x, y, w, h = rectangle
            topLeft = (x, y) # top-left corner
            bottomRight = (x + w, y + h) # bottom-right corner
            cv2.rectangle(image, topLeft, bottomRight, DEFAULT_COLOR)

    def props_filter(self, properties = []):
        """
        Filters out convex hulls of an image by given properties.
        """
        for prop in properties:
            prop_filter = getattr(self, prop + '_filter')
            prop_filter()
        self.rectangles = self.hulls_to_rectangles()

    def aspect_ratio_filter(self):
        """ Filters out convex hulls of an image by aspect ratio. """
        self.hulls = [ hull for hull in self.hulls
                        if aspect_ratio(hull) < ASPECT_RATIO_THRESHOLD ]

    def extent_filter(self):
        """ Filters out convex hulls of an image by extent. """
        self.hulls = [ hull for hull in self.hulls
                        if extent(hull) < LOW_EXTENT_THRESHOLD
                        or extent(hull) > HIGH_EXTENT_THRESHOLD ]

    def solidity_filter(self):
        """ Filters out convex hulls of an image by solidity. """
        self.hulls = [ hull for hull in self.hulls
                        if solidity(hull) < SOLIDITY_THRESHOLD ]

    def SWV_filter(self):
        """ Filters out convex hulls of an image by stroke width variation. """
        edges = [ stroke_width_variation(hull) for hull in self.hulls ]
        self.hulls = [ hull for hull, edge in zip(self.hulls, edges)
                        if SWV_metric(edge) > STROKE_WIDTH_THRESHOLD
                        or math.isnan( SWV_metric(edge) ) ]

    def group_regions(self, threshold = GROUP_THRESHOLD):
        """
        Groups the present rectangles with similar sizes and similar locations.
        Package cv2 function also returns weight (w) of grouped rectangles;
        however, those values are of no use in current program, and are ignored.
        """
        futureRectangles, w = cv2.groupRectangles(self.rectangles, threshold)
        # sanity check in case cv2 grouping removes all rectangles
        if len(futureRectangles) != 0:
            self.rectangles = futureRectangles

    # TODO: add description
    def resize_digits(self, digits, width = 8, height = 8):
        images = []
        for i, d in enumerate(digits):
            x, y, w, h = d
            # scaling a digit to MNIST-based proportion
            x, w = x - 10, w + 20
            # extract a digit from an image
            digit = self.image[y:(y + h), x:(x + w)]
            image = cv2.resize( digit, (width, height) )
            images.append( np.invert(image) )
        return np.array(images)

    def plot_results(self, digits, predictions):
        digits_and_predictions = list(zip(digits, predictions))
        for index, (image, label) in enumerate(digits_and_predictions):
            plt.subplot(2, 4, index + 5)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Prediction: %i' % label)
        plt.show()

    # TODO: add description
    def predict(self, digits, values, clf = classifier.classifier):
        digits = self.resize_digits(digits)
        n_samples = len(digits)
        data = digits.ravel().reshape( (n_samples, -1) )

        expected = values
        predicted = clf.predict(data)

        if len(digits) > 4:
            digits = digits[:4]

        self.plot_results(digits, predicted)

        return (predicted, predicted == expected)
