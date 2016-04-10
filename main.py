# ------------------------------------------------------------------------------
# Author:   Tologon Eshimkanov (https://github.com/tologon)
# Course:   COMP 3770-01 - Introduction to Artificial Intelligence
# School:   Wentworth Institute of Technology
# Project:  Simplified Digit Recognition
# ------------------------------------------------------------------------------

# required package(s)
import argparse, sys
from pipeline import Pipeline

if __name__ == "__main__":
    image, pipeline = None, None
    if len(sys.argv) > 1:
        image = str(sys.argv[1])
        pipeline = Pipeline(image)
    else:
        pipeline = Pipeline()
    pipeline.detect_regions()

    print "before any changes, original rectangles: {}".format( len(pipeline.rectangles) )

    # pipeline.draw_results(pipeline.hulls)
    # pipeline.draw_results(pipeline.contours)
    pipeline.draw_results(pipeline.rectangles)

    properties = ['aspect_ratio', 'extent', 'solidity', 'SWV']
    pipeline.props_filter(properties)
    print "after properties' filters, rectangles: {}".format( len(pipeline.rectangles) )
    pipeline.draw_results(pipeline.rectangles)

    pipeline.group_regions()
    print "after grouping regions, rectangles: {}".format( len(pipeline.rectangles) )
    pipeline.draw_results(pipeline.rectangles)

    result, correctness = pipeline.predict(pipeline.rectangles, [2, 6, 9])
    print "the predicted value of digits: {}".format(result)
    print "does the predicted values matches the actual values? {}".format(correctness)
