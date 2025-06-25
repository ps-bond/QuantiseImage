# Technical Documentation: Image to Vector Converter

This document provides a technical overview of the application's architecture, core algorithms, and data flow. It is intended for developers looking to understand or extend the codebase.

---

## 1. Project Structure

The application is contained within a single main script:

- **`vectorBoundaries.py`**: The main application entry point. It contains:
  - All core processing functions (`process_image_quantisation`, `create_raster_boundaries`, etc.).
  - The main application class (`BoundaryDetectorApp`) which defines the GUI and its logic.
  - Custom widget classes (`ResizableImageLabel`, `CustomTickSlider`).
  - The `Worker` class for handling background processing.

---

## 2. GUI Architecture

- **Main Window**: The `BoundaryDetectorApp` class inherits from `QMainWindow` to provide a standard application window with a menu bar. All other UI elements are placed within a central widget.

- **Responsiveness & Threading**:
  - The most computationally expensive operation, K-Means clustering, is offloaded to a `QThread` via the `Worker` class.
  - The `Worker`'s `run` method performs the quantisation and emits a `finished` signal with the resulting `label_grid` upon completion.
  - This prevents the GUI from freezing during analysis. The main thread listens for the `finished` signal and updates the UI accordingly.

- **Dynamic Controls**:
  - **Colour Slider**: The `sliderReleased` signal is connected to `start_processing`. This ensures the expensive K-Means algorithm is only re-run when the user has finished selecting a new colour value.
  - **Smoothing Slider**: The `valueChanged` signal is connected to `_update_smoothed_preview`. Since the median filter is much faster than K-Means, it can be applied in real-time as the slider moves, providing an interactive user experience.

- **Custom Widgets**:
  - **`ResizableImageLabel`**: A `QLabel` subclass that overrides `resizeEvent`. It scales its internal `QPixmap` to fit the widget's current size while maintaining the aspect ratio. Its size policy is set to `Expanding` to ensure it grows to fill available layout space.
  - **`CustomTickSlider`**: A `QSlider` subclass that overrides `paintEvent` to draw tick marks at specific, non-linear positions. This is necessary because the default `QSlider` only supports uniformly spaced ticks.

---

## 3. Core Algorithms & Data Flow

The processing pipeline follows these steps:

1.  **Image Loading (`_open_file`)**:
    -   An image file is opened using `Pillow` and converted to an RGB `numpy` array (`self.input_img_data`).
    -   The application state is reset, and `start_processing` is called automatically.

2.  **Colour Quantisation (`process_image_quantisation`)**:
    -   This function is called within the `Worker` thread.
    -   The image's pixel data is reshaped into a 2D array (list of pixels).
    -   `sklearn.cluster.KMeans` is used to find `n` cluster centers in the 3D RGB space. These centers become the new, limited colour palette.
    -   The `ConvergenceWarning` is suppressed, as it's expected and harmless when an image has fewer unique colours than the requested number of clusters.
    -   The function returns a 2D `numpy` array (`label_grid`) where each element is the integer index of the new colour for that pixel.

3.  **Smoothing (`_update_smoothed_preview`)**:
    -   The raw `label_grid` from the worker is stored as `self.quantised_label_grid`.
    -   The integer value (0-100) from the smoothing slider is converted to a float (0.0-10.0).
    -   If smoothing is enabled, `scipy.ndimage.median_filter` is applied to `self.quantised_label_grid`. The kernel size for the filter is calculated from the float smoothing level, ensuring it is always an odd integer.
    -   The result is stored in `self.smoothed_label_grid`.

4.  **Boundary Detection & Preview (`create_raster_boundaries`)**:
    -   This function takes the (potentially smoothed) `label_grid`.
    -   It uses efficient, vectorised `numpy` operations to find boundaries. It creates boolean masks by comparing each pixel's label with its right-hand (`label_grid[:, :-1] != label_grid[:, 1:]`) and bottom (`label_grid[:-1, :] != label_grid[1:, :]`) neighbours.
    -   A new greyscale image array is created, and the boundary pixels are set to black (0). This array is used for the GUI preview.

5.  **Vectorisation (`generate_svg_boundaries`)**:
    -   This function is called only during a save operation (`_save_file` or `_save_file_as`).
    -   It uses the pre-calculated `horiz_diff_map` and `vert_diff_map` to identify contiguous horizontal and vertical line segments.
    -   It then combines these segments into single `<path>` elements (using SVG's `M` for "move to" and `L` for "line to" commands) to reduce the number of elements in the SVG output.
    -   The final drawing is saved to the specified `.svg` file.

- **PNG Output**: A simple function `generate_png_boundaries` uses `Pillow` to save the greyscale boundary `numpy` array directly as a PNG image.
---

## 4. Potential Future Enhancements

- **SVG Optimisation**: The current `generate_svg_boundaries` function creates a large number of small, individual line segments. A significant optimisation would be to implement a path-finding algorithm to merge adjacent, co-linear segments into longer `<path>` elements, which would drastically reduce the SVG file size and complexity.
- **SVG Optimisation (Further)**: While adjacent lines are combined, more advanced algorithms could connect non-adjacent but logically continuous paths (e.g., around corners) into single, more complex SVG paths.
- **Batch Processing**: Implement a feature to process an entire directory of images from the command line.