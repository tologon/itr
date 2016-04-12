Simplified Digit Recognition
============================

The simplified digit recognition is a tool that helps to extract
digit(s) from an image without a lot of background noises.

Dependencies
------------
* Python 2.7
* Python libraries:
    * [NumPy][numpy]
    * [OpenCV][openCV]
    * [scikit-learn][sklearn]
    * [matplotlib][matplotlib]

*WARNING*: Before you proceed, it should be known that this script was tested
only on **Mac operating system**. It might work on **Ubuntu operating system**
as well. The script might probably not work on **Windows operating system**.

To correctly install OpenCV, please follow this [guide][openCV_guide].

Once you've followed OpenCV guide, use ``` pip ```
command to install the remaining Python libraries:
```
pip install numpy
pip install sklearn
pip install matplotlib
```

Information
-----------
The simplified digit recognition consists of multiple steps,
described as following pipeline:

1. Detect candidate digit regions using Maximally Stable Extremal Regions (MSER).
1. Remove non-digit regions based on geometric properties
(such as aspect ratio, extent, and solidity).
1. Remove non-digit regions based on stroke width variation (stroke width is a measure of the width of the curves and lines that make up a character; non-digit regions tend to have large variations).
1. Group the candidate regions by their proximity.
1. Recognize and detect digit(s).

The last step is achieved by the following machine learning algorithm:

* Algorithm - Support Vector Machine
* Kernel (separates & classifies the data) - Linear kernel

Usage
-----
All the work is being processed in **main.py** and then delegated to
other files.

To execute the script, simply run the following command:
```
python main.py
```

To get useful info about the script, simply pass ``` --help ```
or ``` -h ``` flag to the command:
```
python main.py --help
```

To perform digit(s) recognition on default images (single or multiple digits),
add ``` --example ``` or ``` -e ``` flag with either
option **single** or **multiple**:
```
python main.py -e single
```

The rest of useful commands can be invoked on
**main.py** script with the passed ``` --help ``` flag.



[openCV_guide]: http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/
[numpy]: http://www.numpy.org/
[openCV]: http://opencv.org/
[sklearn]: http://scikit-learn.org
[matplotlib]: http://matplotlib.org/
