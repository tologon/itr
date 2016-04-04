# ------------------------------------------------------------------------------
# Author:   Tologon Eshimkanov (https://github.com/tologon)
# Course:   COMP 3770-01 - Introduction to Artificial Intelligence
# School:   Wentworth Institute of Technology
# Project:  Image Text Recognition
# ------------------------------------------------------------------------------

# required package(s)
import sys, cv2, math, mnist
import numpy as np

from properties import aspectRatio, extent, solidity, strokeWidthVariation, strokeWidthMetric
from mnist import plt

# ignore sys.argv[0] as it is a name of an invoked Python script
# print 'Number of arguments:', len(sys.argv[1:]), 'arguments.'
# print 'Argument list:', str(sys.argv[1:])

# constants
DEFAULT_SINGLE_DIGIT = 'default_single_digit.png'
DEFAULT_COLOR = (0, 255, 0) # RGB values; doesn't matter on grayscale
ASPECT_RATIO_THRESHOLD = 0.29
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
        self.hulls = self.regionsToHulls() # hulls == convex hulls
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

    def drawResults(self, results):
        """ Draws given results on the original image. """
        imageCopy = self.image.copy()
        isClosed = 1 # no documentation available
        # the results are in a form of rectangles
        if len(results[0]) == 4:
            self.drawRectangles(imageCopy, results)
        # the results are in a form of convex hulls
        else:
            cv2.polylines(imageCopy, results, isClosed, DEFAULT_COLOR)
        cv2.imshow('Results', imageCopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def drawRectangles(self, image, rectangles):
        """ Draws rectangles on the image. """
        for rectangle in rectangles:
            x, y, w, h = rectangle
            topLeftCorner = (x, y)
            bottomRightCorner = (x + w, y + h)
            # print "top left corner: {} | bottom right corner: {}".format(topLeftCorner, bottomRightCorner)
            cv2.rectangle(image, topLeftCorner, bottomRightCorner, DEFAULT_COLOR)

    def filterByProperties(self, properties = []):
        """
        Filters out convex hulls of an image by given properties.
        """
        for prop in properties:
            # print "looping in properties"
            filterByProperty = getattr(self, 'filterBy' + prop)
            filterByProperty()

    # TODO: refactor for readability
    def filterByAspectRatio(self):
        """ Filters out convex hulls of an image by aspect ratio. """
        self.hulls = [hull for hull in self.hulls if aspectRatio(hull) > ASPECT_RATIO_THRESHOLD]

    # TODO: refactor for readability
    def filterByExtent(self):
        """ Filters out convex hulls of an image by extent. """
        self.hulls = [hull for hull in self.hulls if extent(hull) < LOW_EXTENT_THRESHOLD or extent(hull) > HIGH_EXTENT_THRESHOLD]

    # TODO: refactor for readability
    def filterBySolidity(self):
        """ Filters out convex hulls of an image by solidity. """
        self.hulls = [hull for hull in self.hulls if solidity(hull) < SOLIDITY_THRESHOLD]

    # TODO: refactor for readability
    def filterByStrokeWidthVariation(self):
        """ Filters out convex hulls of an image by stroke width variation. """
        edges = [strokeWidthVariation(hull) for hull in self.hulls]
        self.hulls = [hull for hull, edge in zip(self.hulls, edges) if strokeWidthMetric(edge) > STROKE_WIDTH_THRESHOLD or math.isnan(strokeWidthMetric(edge))]

    def groupRegions(self, threshold = GROUP_THRESHOLD):
        """
        Groups the present rectangles with similar sizes and similar locations.
        Package cv2 function also returns weight of each grouped rectangle;
        however, those values are of no use in current program, and are ignored.
        """
        (self.rectangles, weights) = cv2.groupRectangles(self.rectangles, threshold)

    # TODO: add description
    def predict(self, digits, values, classifier = mnist.classifier):
        width, height = 8, 8
        images = []
        for i, d in enumerate(digits):
            x, y, w, h = d
            x, w = x - 10, w + 20 # scaling digit to MNIST-based size
            digit = self.image[y:(y + h), x:(x + w)]
            # cv2.imwrite('test-' + str(i) + '.png', digit)
            image = cv2.resize( digit, (width, height) )
            images.append( np.invert(image) )
            # images.append(image)

        images = np.array(images)
        n_samples = len(images)
        data = images.ravel().reshape( (n_samples, -1) )
        # print "data shape: {}".format(data.shape)

        expected = values
        predicted = classifier.predict(data)

        images_and_predictions = list(zip(images, predicted))
        for index, (image, label) in enumerate(images_and_predictions):
            plt.subplot(2, 4, index + 5)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Prediction: %i' % label)
        plt.show()

        return (predicted, predicted == expected)

if __name__ == "__main__":
    image, pipeline = None, None
    if len(sys.argv) > 1:
        image = str(sys.argv[1])
        pipeline = Pipeline(image)
    else:
        pipeline = Pipeline()
    pipeline.detectRegions()

    print "before any changes, original rectangles: {}".format( len(pipeline.rectangles) )

    # pipeline.drawResults(pipeline.hulls)
    # pipeline.drawResults(pipeline.contours)
    pipeline.drawResults(pipeline.rectangles)

    properties = ['AspectRatio', 'Extent', 'Solidity', 'StrokeWidthVariation']
    pipeline.filterByProperties(properties)
    print "after properties' filters, rectangles: {}".format( len(pipeline.rectangles) )
    pipeline.drawResults(pipeline.rectangles)

    pipeline.groupRegions()
    print "after grouping regions, rectangles: {}".format( len(pipeline.rectangles) )
    pipeline.drawResults(pipeline.rectangles)

    result, correctness = pipeline.predict(pipeline.rectangles, [2, 6, 9])
    print "the predicted value of digits: {}".format(result)
    print "does the predicted values matches the actual values? {}".format(correctness)
