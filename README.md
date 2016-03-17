**Image Text Recognition**

*WARNING*: For the Python scripts to work, a very specific configuration of Python libraries is required. Such configuration will be added later.

The image text recognition consists of multiple steps,
described as following (the initial pipeline consists of steps 1 – 3):

1. Detect candidate text regions using Maximally Stable Extremal Regions (MSER).
2. Remove non-text regions based on geometric properties (such as aspect ratio, eccentricity, Euler number, extent, and solidity).
3. Remove non-text regions based on stroke width variation (stroke width is a measure of the width of the curves and lines that make up a character; non-text regions tend to have large variations).
4. Recognize and detect text.

The last step (4) can be achieved by a couple of different approaches:

* Given text regions, group them by their proximity and recognize text using text classifier (individual character classifier, or otherwise known as Optical Character Recognition, or OCR).
* Given text regions, perform additional text detection (using text & no text classifier), followed by text classification (individual character classifier), ending with the text reconstruction (grouping close text together).
