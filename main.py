# ------------------------------------------------------------------------------
# Author:   Tologon Eshimkanov (https://github.com/tologon)
# Course:   COMP 3770-01 - Introduction to Artificial Intelligence
# School:   Wentworth Institute of Technology
# Project:  Simplified Digit Recognition
# ------------------------------------------------------------------------------

# required package(s)
import argparse
from pipeline import Pipeline

# constants
DEFAULT_SINGLE_DIGIT = 'default_single_digit.png'
DEFAULT_COLOR = (0, 255, 0) # RGB values; doesn't matter on grayscale
ASPECT_RATIO_THRESHOLD = 0.5
LOW_EXTENT_THRESHOLD = 0.3
HIGH_EXTENT_THRESHOLD = 0.59
SOLIDITY_THRESHOLD = 1.1
STROKE_WIDTH_THRESHOLD = 0.99
GROUP_THRESHOLD = 1 # minimum possible number of rectangles - 1

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
