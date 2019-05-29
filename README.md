# Background Zero
Extraction of region of interest from an image. Adaptation of a median-calculation code developed by Sergio Orts-Escolano (University of Alicante).

## What the code does
It recieves two images, the first one with just a background and the second one with background and something on the foreground.
It outputs a black-and-white image with white being the foreground black being the background.

The code runs two versions of the algorithm, with and without parallelization.
It compares it's execution times with and without the data transfer time between host and device.

## Execution parameters
0. Executable name, as always
1. Image with only background (if using the supplied project, this is Images/background.bmp)
2. Image with foreground elements (if using the supplied project, this is Images/image.bmp)
3. Threshhold value for the algorithm to use. A typical value is 150.

## How the algorithm works
For every pixel in the foreground image: 
    Calculates the mean value (in grey-scale) for the 9-pixel cluster surrounding it.
    Performs the subtraction of said mean minus the value of the same pixel in the background image.
    The pixel is set as white or black depending on wether the absolute value of said subtraction is above or below the given threshhold.
