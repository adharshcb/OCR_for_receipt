# Optical Character Recognition (OCR) for Receipts

This project is an implementation of OCR for receipts using Python, OpenCV, and PyTesseract. It extracts text from an image of a receipt by identifying specific regions of interest (ROI) on the image and using Tesseract OCR to read the text within these ROIs.

### Installation
To install and use this project, first create a new Conda environment using the provided requirements.txt file:

```
conda create --name <env> --file requirements.txt
```
This will install all the necessary dependencies.

### Usage
To use the project, run the main.py script. This will process all the images in the images directory and output the extracted text for each ROI. The script also displays visualizations of the processed images, including matched keypoint features, homography transformations, and cropped ROIs.

### Dependencies
The project uses the following dependencies:

Python 3.10
OpenCV
NumPy
PyTesseract
imutils

### Contributing
Contributions to this project are welcome. I am just a noob in cv2. If you can make make any improvement and contribute code, please fork the repository and create a pull request.

