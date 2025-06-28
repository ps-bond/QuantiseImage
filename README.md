# Image to Vector Converter

**Version:** 0.1

A desktop application built with Python and PyQt6 that analyses an image, quantises its colours, and generates a clean, scalable vector (SVG) file of the boundaries between colour regions.

![Application Screenshot](/images/exampleWindow.png)

---

## Features

- **Interactive GUI**: Easy-to-use graphical interface built with PyQt6.
- **Colour Quantisation**: Reduces the image's colour palette to a user-specified number of colours using the K-Means algorithm.
- **Live Preview**: Instantly see a preview of the detected boundaries.
- **Dynamic Controls**:
  - Adjust the number of colours and see the result after a quick re-analysis.
  - Apply a smoothing filter in real-time with a continuous slider to reduce noise and simplify the output.
- **Vector Output**: Saves the final boundaries as a scalable SVG file, perfect for graphic design and further editing.
- **Raster Output**: Also supports saving the boundaries as a PNG image.
- **Standard Interface**:
  - Fully resizable window with scaling image previews.
  - Standard File menu (`Open`, `Save`, `Save As...`, `Exit`).
  - Help menu with an "About" dialog.

## Installation and Usage

There are two ways to use this application.

### Option 1: Pre-built Executable (Recommended for Windows users)

1.  Download the `vectorBoundaries.exe` from the project's release page.
2.  Double-click the file to run. No installation is required.

### Option 2: From Source Code (For all platforms)

#### 1. Requirements

This project requires Python 3 and the following libraries:

- `PyQt6`
- `Pillow`
- `numpy`
- `scikit-learn`
- `scipy`
- `svgwrite`

#### 2. Installation

1.  **Clone the repository or download the source code.**
2.  **Install the required dependencies using pip:**

    ```bash
    pip install -r requirements.txt
    ```

#### 3. Usage

1.  **Run the application from your terminal:**

    ```bash
    python vectorBoundaries.py
    ```

    Or run
    ```
    vectorBoundaries.exe
    ```
    on Windows.

    
## Application Guide

1.  **Open an Image**: Use the `File > Open...` menu to select an input image (`.png`, `.jpg`, etc.). The application will automatically perform an initial analysis.

2.  **Adjust Settings**:
    -   Use the **"Number of Colours"** slider to change the level of detail in the quantisation. Release the slider to trigger a new analysis.
    -   Use the **"Smoothing Level"** slider to dynamically apply a median filter. This helps remove noise and create smoother boundary lines.

3.  **Save the Output**: Use the `File > Save` or `File > Save As...` menu to save the resulting boundaries as an `.svg` or a `.png` file.

## How It Works

1.  **Colour Quantisation**: The application loads an image and uses the K-Means clustering algorithm from `scikit-learn` to group all the pixel colours into a smaller, representative palette. Each pixel is then assigned a label corresponding to one of the new colours.

2.  **Smoothing (Optional)**: A median filter from `scipy` is applied to the grid of colour labels. This process effectively removes small, isolated pixels ("salt-and-pepper noise") and smooths the edges between larger colour regions.

3.  **Boundary Detection**: Using `numpy` for high performance, the code compares each pixel's label to its neighbours. A boundary is detected wherever two adjacent pixels have different labels.

4.  **Vectorisation**: The detected pixel-level boundaries are translated into a series of 1-pixel-long horizontal and vertical lines and written to an SVG file using the `svgwrite` library.
