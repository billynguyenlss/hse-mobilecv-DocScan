Document Scanner

This project under the course "Computer Vision for mobile" from HSE MCV program. This project included:

* Select single image from Gallery and load to UI
* Capture image from Camera and save with original resolution
* Select video from Gallery and select any interested frame. Please do following steps:
  * Select video from Gallery
  * Click on `Select Frame` to select interested frame.
  * Once selected your interested frame, click on `Close video` to disable the Select Frame from Video feature.
* Processing operators:
  * Binarization: (similar to provided code from previous week)
  * Image filtering: (Gaussian Filter for demonstration, similar to provided code from previous week. Other filters (`bilateral`, `fft`) are available not show in the Toolbar).
  * Noise removal: (cv::Median filter for simple Noise Removal.)
  * Contrast enhancement
* Stitching 2 images: similar to provided code from week 5
* Stitching multiple images: native C++ function from OpenCV is implemented
* Manually select 4 corners for geometric transformation: similar to provided code from week 5
* Auto select 4 corners for geometric transformation: implemented, with reference from https://learnopencv.com/automatic-document-scanner-using-opencv/
* Implementation of algorithms outside of OpenCV `char_threshold`, reference from https://www.mvtec.com/doc/halcon/13/en/char_threshold.html

Please take a look on the video `DocScanDemo.mp4` to see how the application work.
