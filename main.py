# ------------------------------------------------------------------------------
# Author:   Tologon Eshimkanov (https://github.com/tologon)
# Course:   COMP 3770-01 - Introduction to Artificial Intelligence
# School:   Wentworth Institute of Technology
# Project:  Simplified Digit Recognition
# ------------------------------------------------------------------------------

# required package(s)
import argparse
from pipeline import Pipeline, DEFAULT_SINGLE_DIGIT, DEFAULT_MULTIPLE_DIGIT

# TODO: add description
def parse_options():
    parser = argparse.ArgumentParser(description="recognize digit(s) in image",
                                        add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-i", "--image", type=str,
                        help="image containing digit(s)")
    group.add_argument("-e", "--example", choices=["single", "multiple"],
                         type=str, help="use one of default images")
    parser.add_argument("-d", "--digits", nargs="+",
                        type=int, help="actual values of digit(s) in the image")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-h", "--help", action="help",
                        help="display information about script")
    return parser.parse_args()

# TODO: add description
def initialize_pipeline():
    pipeline = None
    if args.image:
        pipeline = Pipeline(args.image)
    elif args.example:
        if args.example == 'single':
            pipeline = Pipeline(DEFAULT_SINGLE_DIGIT)
        elif args.example == 'multiple':
            pipeline = Pipeline(DEFAULT_MULTIPLE_DIGIT)
    else:
        pipeline = Pipeline()
    return pipeline

# TODO: add description
def output(message, pipeline):
    if args.verbose:
        num_rectangles = len(pipeline.rectangles)
        print "{}, # of rectangles: {}".format(message, num_rectangles)
        pipeline.draw_results( pipeline.rectangles )

args = parse_options()

if __name__ == "__main__":
    pipeline = initialize_pipeline()

    pipeline.detect_regions()
    output("before changes", pipeline)

    properties = ['aspect_ratio', 'extent', 'solidity', 'SWV']
    pipeline.props_filter(properties)
    output("after filters", pipeline)

    pipeline.group_regions()
    output("after grouping", pipeline)

    result = pipeline.predict(pipeline.rectangles)
    print "the predicted value(s) of digit(s): {}".format(result)
    if args.digits:
        print "correctness of prediction(s): {}".format(result == args.digits)
