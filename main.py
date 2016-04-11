# ------------------------------------------------------------------------------
# Author:       Tologon Eshimkanov (https://github.com/tologon)
# Course:       COMP 3770-01 - Introduction to Artificial Intelligence
# School:       Wentworth Institute of Technology
# Project:      Simplified Digit Recognition
# Description:
#   This file is the one that a user will invoke. It involves parsing
#   of command-line options, pipeline initialization and output display.
# ------------------------------------------------------------------------------

# required package(s)
import argparse
from pipeline import *

def parse_options():
    """
    Parses and processes the command-line supplied by a user.
    usage:
    main.py [-i IMAGE | -e {single,multiple}] [-d DIGITS [DIGITS ...]] [-v] [-h]

    recognize digit(s) in image

    optional arguments:
      -i, --image IMAGE                 image containing digit(s)
      -e, --example {single,multiple}   use one of default images
      -d, --digits DIGITS [DIGITS ...]  actual values of digit(s) in the image
      -v, --verbose                     increase output verbosity
      -h, --help                        display information about script
    """
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

def initialize_pipeline():
    """
    Given command-line options, creates a pipeline
    with an image (provided or supplied by default).
    """
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

def output(message, pipeline):
    """ Given verbose option, prints & draws results to the user. """
    if args.verbose:
        num_rectangles = len(pipeline.rectangles)
        print "{}, # of rectangles: {}".format(message, num_rectangles)
        pipeline.draw_results( pipeline.rectangles )

args = parse_options()

if __name__ == "__main__":
    """
    Executes the entire pipeline for simplified digit(s) recognition.
    Such pipeline includes the following:
        - pipeline initialization
        - MSER regions detection
        - regions filtering by properties (geometric & text-based)
        - regions grouping by their proximity
        - cross-validation of MNIST training data
        - digit(s) recognition
        - output results (regular or verbose) to a user
        - (optional) correctness of the results
    """
    pipeline = initialize_pipeline()

    pipeline.detect_regions()
    output("before changes", pipeline)

    properties = ['aspect_ratio', 'extent', 'solidity', 'SWV']
    pipeline.props_filter(properties)
    output("after filters", pipeline)

    pipeline.group_regions()
    output("after grouping", pipeline)

    pipeline.cross_validate(k_fold = 10)

    result = pipeline.predict(pipeline.rectangles)

    print "predicted value(s) of digit(s) for {}: {}".format(
        pipeline.image_name, result)

    if args.digits:
        print "correctness of prediction(s): {}".format(result == args.digits)
    elif args.example:
        if args.example == 'single':
            print "correctness of prediction(s): {}".format(result == [5])
        elif args.example == 'multiple':
            print "correctness of prediction(s): {}".format(result == [2, 6, 9])
