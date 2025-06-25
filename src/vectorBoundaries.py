import sys
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from typing import Tuple
import svgwrite
from scipy.ndimage import median_filter
import warnings

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QFormLayout, QSlider, QStyle, QStyleOptionSlider, QSizePolicy,
    QMainWindow, QMessageBox, QMenu
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt
from PyQt6.QtGui import QPixmap, QImage, QPainter, QAction


__version__ = "0.1"


def process_image_quantisation(img_data: np.ndarray, num_colours: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantises an image and returns the quantised image and the label grid.

    Args:
        img_data (np.ndarray): Input image data in RGB format (height, width, 3).
        num_colours (int): The number of colours to quantise the image to.

    Returns:
        A tuple of (quantised_image_data, label_grid).
    """
    # Reshape the image data to be a list of pixels (N_pixels, 3)
    pixels = img_data.reshape(-1, 3)

    # Use KMeans to find the most representative colours
    kmeans = KMeans(n_clusters=num_colours, n_init='auto', random_state=42)

    # Suppress ConvergenceWarning when the number of unique colours is less than n_clusters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        kmeans.fit(pixels)

    # Create the quantised image
    new_colours = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    quantised_pixels = new_colours[labels]
    quantised_image_data = quantised_pixels.reshape(img_data.shape).astype(np.uint8)

    # Reshape the labels to match the original image dimensions
    label_grid = labels.reshape(img_data.shape[0], img_data.shape[1])

    return quantised_image_data, label_grid


def create_raster_boundaries(label_grid: np.ndarray) -> np.ndarray:
    """
    Creates a raster (pixel) image of boundaries from a label grid for preview.

    Args:
        label_grid (np.ndarray): A 2D array where each value is a colour index.

    Returns:
        np.ndarray: A greyscale image array where 0 is a boundary and 255 is not.
    """
    # Create a blank (white) canvas for the output
    boundaries_data = np.full(label_grid.shape, 255, dtype=np.uint8)

    # Find horizontal boundaries using vectorized numpy operations for speed
    horizontal_diff = label_grid[:, :-1] != label_grid[:, 1:]
    boundaries_data[:, :-1][horizontal_diff] = 0

    # Find vertical boundaries
    vertical_diff = label_grid[:-1, :] != label_grid[1:, :]
    boundaries_data[:-1, :][vertical_diff] = 0

    return boundaries_data


def generate_svg_boundaries(label_grid: np.ndarray, output_svg_path: str):
    """Generates an SVG file with 1-unit thick lines from a label grid."""
    """Generates an SVG file with 1-unit thick lines from a label grid,
    optimising by combining adjacent lines into longer paths."""
    height, width = label_grid.shape
    dwg = svgwrite.Drawing(output_svg_path, size=(f"{width}px", f"{height}px"), profile='tiny')

    # Pre-calculate differences for efficiency
    # horiz_diff_map[y, x] is true if a vertical line exists between (x,y) and (x+1,y)
    # The line segment is from (x+1, y) to (x+1, y+1)
    horiz_diff_map = (label_grid[:, :-1] != label_grid[:, 1:]).astype(bool)
    # vert_diff_map[y, x] is true if a horizontal line exists between (x,y) and (x,y+1)
    # The line segment is from (x, y+1) to (x+1, y+1)
    vert_diff_map = (label_grid[:-1, :] != label_grid[1:, :]).astype(bool)

    # Create visited maps to avoid re-processing segments
    visited_horiz = np.zeros_like(horiz_diff_map, dtype=bool)
    visited_vert = np.zeros_like(vert_diff_map, dtype=bool)

    # --- Optimise Vertical Lines (from horiz_diff_map) ---
    # These are lines at x+1, extending vertically from y
    for y in range(height):
        for x in range(width - 1): # Iterate up to width-2 for horiz_diff_map
            if horiz_diff_map[y, x] and not visited_horiz[y, x]:
                # Start of a new vertical segment
                start_y = y
                current_y = y
                
                # Extend downwards
                # Check if the boundary continues at the same x-coordinate in the next row
                while current_y < height and horiz_diff_map[current_y, x] and not visited_horiz[current_y, x]:
                    visited_horiz[current_y, x] = True
                    current_y += 1
                
                end_y = current_y # The end y-coordinate of the path

                # Add the vertical path
                path_data = f"M {x + 1} {start_y} L {x + 1} {end_y}"
                dwg.add(dwg.path(d=path_data, stroke='black', stroke_width=1, fill='none'))

    # --- Optimise Horizontal Lines (from vert_diff_map) ---
    # These are lines at y+1, extending horizontally from x
    for y in range(height - 1): # Iterate up to height-2 for vert_diff_map
        for x in range(width):
            if vert_diff_map[y, x] and not visited_vert[y, x]:
                # Start of a new horizontal segment
                start_x = x
                current_x = x

                # Extend rightwards
                # Check if the boundary continues at the same y-coordinate in the next column
                while current_x < width and vert_diff_map[y, current_x] and not visited_vert[y, current_x]:
                    visited_vert[y, current_x] = True
                    current_x += 1
                
                end_x = current_x # The end x-coordinate of the path

                # Add the horizontal path
                path_data = f"M {start_x} {y + 1} L {end_x} {y + 1}"
                dwg.add(dwg.path(d=path_data, stroke='black', stroke_width=1, fill='none'))

    dwg.save()


def generate_png_boundaries(label_grid: np.ndarray, output_png_path: str):
    """Generates a PNG file from a greyscale raster image data.
    The raster_data is expected to be a NumPy array (0=black, 255=white).
    """
    # Ensure data is uint8 and is a contiguous array (a copy if not already)
    img_array = np.ascontiguousarray(label_grid, dtype=np.uint8)
    boundary_image = Image.fromarray(img_array, mode='L') # 'L' for greyscale
    boundary_image.save(output_png_path)


class Worker(QObject):
    """
    A worker object that runs the image processing in a separate thread
    to prevent the GUI from freezing.
    """
    finished = pyqtSignal(object)  # Will emit (label_grid)
    progress = pyqtSignal(str)

    def __init__(self, img_data, num_colours): # Removed smoothing_level
        super().__init__()
        self.img_data = img_data
        self.num_colours = num_colours

    def run(self):
        self.progress.emit("Quantising colours...")
        _, label_grid = process_image_quantisation(self.img_data, self.num_colours)

        self.finished.emit(label_grid) # Only emit the raw label_grid


class ResizableImageLabel(QLabel):
    """A QLabel subclass that automatically resizes its pixmap, maintaining aspect ratio."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)  # A sensible minimum size for resizability
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def setPixmap(self, pixmap: QPixmap):
        """Sets the internal pixmap and triggers a display update."""
        self._pixmap = pixmap
        self._update_pixmap()

    def resizeEvent(self, event):
        """Handle the resize event to scale the pixmap."""
        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self):
        """Scales the internal pixmap to fit the label's current size."""
        if self._pixmap.isNull():
            super().setPixmap(QPixmap())  # Clear the pixmap
            return

        scaled_pixmap = self._pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)


class CustomTickSlider(QSlider):
    """A QSlider subclass that allows for custom, non-uniform tick positions."""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._tick_positions = []

    def setTickPositions(self, positions: list[int]): # type: ignore
        """Sets the list of integer values where tick marks should be drawn."""
        self._tick_positions = positions
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        # Draw the groove first.
        opt.subControls = QStyle.SubControl.SC_SliderGroove
        self.style().drawComplexControl(QStyle.ComplexControl.CC_Slider, opt, painter, self)

        # Draw the custom tick marks.
        groove_rect = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self)
        tick_y = groove_rect.y() + groove_rect.height() + 2
        if self._tick_positions:
            for value in self._tick_positions:
                if self.minimum() <= value <= self.maximum():
                    # Calculate the x-position for the tick mark by mapping the logical value to a pixel coordinate
                    x = self.style().sliderPositionFromValue(self.minimum(), self.maximum(), value, groove_rect.width(), opt.upsideDown) + groove_rect.x()
                    painter.drawLine(x, tick_y, x, tick_y + 5)

        # Draw the handle on top of the groove and ticks.
        opt.subControls = QStyle.SubControl.SC_SliderHandle
        self.style().drawComplexControl(QStyle.ComplexControl.CC_Slider, opt, painter, self)


class BoundaryDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_img_data = None
        self.quantised_label_grid = None # Stores the raw quantised grid from K-Means
        self.smoothed_label_grid = None  # Stores the currently smoothed grid for display/saving (raster preview)
        self.current_raster_boundary_data = None # Stores the actual black/white boundary image data for saving PNG
        self.output_file_path = None # Stores the path for saving (SVG or PNG)
        self.output_file_format = None # Stores the format ('svg' or 'png')
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f'Image to Vector Converter v{__version__}')
        self.setGeometry(100, 100, 1000, 600)

        # Create a central widget to hold all the layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # --- Layouts ---
        main_layout = QVBoxLayout(central_widget)
        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        image_layout = QHBoxLayout()

        # --- Widgets ---
        # Settings
        self.num_colours_slider = CustomTickSlider(Qt.Orientation.Horizontal, self)
        self.num_colours_slider.setRange(2, 64) # Set range from 2 to 64
        self.num_colours_slider.setTickPositions([2, 4, 8, 16, 32, 48, 64])
        self.num_colours_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.num_colours_slider.setValue(8) # Initial value
        
        self.num_colours_label = QLabel(str(self.num_colours_slider.value()), self)
        self.num_colours_slider.valueChanged.connect(lambda value: self.num_colours_label.setText(str(value)))
        self.num_colours_slider.sliderReleased.connect(self.start_processing) # Re-process when slider is released

        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.smoothing_slider.setRange(0, 100) # Scale from 0 to 100 for float simulation (0.0 to 10.0)
        self.smoothing_slider.setValue(0)
        self.smoothing_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.smoothing_slider.setTickInterval(10) # Tick marks at every 1.0 on the float scale

        self.smoothing_label = QLabel(f"{self.smoothing_slider.value() / 10.0:.1f}", self) # Display current smoothing value
        self.smoothing_slider.valueChanged.connect(self._update_smoothed_preview) # Connect for dynamic update

        # Status and Image Display
        self.status_label = QLabel("Please select an input image.", self)
        self.img_display_in = ResizableImageLabel(self)
        self.img_display_out = ResizableImageLabel(self)

        # --- Assemble Layout ---
        num_colours_layout = QHBoxLayout()
        num_colours_layout.addWidget(self.num_colours_slider)
        num_colours_layout.addWidget(self.num_colours_label)
        form_layout.addRow("Number of Colours:", num_colours_layout)

        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(self.smoothing_slider)
        smoothing_layout.addWidget(self.smoothing_label)
        form_layout.addRow("Smoothing Level:", smoothing_layout)

        image_layout.addWidget(self.img_display_in, 1) # Add stretch factor to make it expand
        image_layout.addWidget(self.img_display_out, 1) # Add stretch factor to make it expand

        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(image_layout)

        self._create_menu_bar()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        self.save_action = QAction("&Save...", self) # Changed to Save... to imply dialog might open
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self._save_file)
        self.save_action.setEnabled(False)
        file_menu.addAction(self.save_action)

        self.save_as_action = QAction("Save &As...", self)
        self.save_as_action.triggered.connect(self._save_file_as)
        self.save_as_action.setEnabled(False)
        file_menu.addAction(self.save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help Menu Button on the far right
        # On some platforms (like macOS) this is handled automatically, but for
        # others (like Windows) using a corner widget is the correct approach.
        help_button = QPushButton("&Help", self)
        help_button.setFlat(True) # Makes it look like a menu item instead of a raised button

        help_menu = QMenu(self)
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        help_button.setMenu(help_menu)
        menu_bar.setCornerWidget(help_button)

    def show_about_dialog(self):
        about_text = f"""
        <h2>Image to Vector Converter</h2>
        <p>Version {__version__}</p>
        <p>This application analyses an image, quantizes it into a user-specified number of colours, and creates a vector (SVG) output that consists only of the boundaries of each colour region.</p>
        <p>Developed by Peter, with assistance from Gemini Code Assist.</p>
        """
        QMessageBox.about(self, "About Image to Vector Converter", about_text)

    def _open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            try:
                with Image.open(file_path) as img:
                    img_rgb = img.convert("RGB")
                    self.input_img_data = np.array(img_rgb)
                
                # Reset state
                self.output_svg_path = None
                self.quantised_label_grid = None # Reset
                self.smoothed_label_grid = None  # Reset
                self.save_action.setEnabled(False)
                self.save_as_action.setEnabled(False)
                self.img_display_out.setPixmap(QPixmap()) # Clear output display

                pixmap = self.numpy_to_qpixmap(self.input_img_data)
                self.img_display_in.setPixmap(pixmap)
                self.status_label.setText(f"Loaded {file_path}. Starting analysis...")
                self.start_processing() # Automatically start processing
            except Exception as e: # Catch any errors during image loading
                self.status_label.setText(f"Error loading image: {e}") # Display error in status bar

    def _save_file(self):
        if self.current_raster_boundary_data is None: # Check for the actual boundary image data
            return # Should not happen if action is disabled
        
        if not self.output_file_path:
            self._save_file_as()
            return

        try: # Attempt to save the file
            self.status_label.setText(f"Saving to {self.output_file_path}...")
            if self.output_file_format == 'svg':
                generate_svg_boundaries(self.smoothed_label_grid, self.output_file_path) # SVG needs the label grid
            elif self.output_file_format == 'png':
                generate_png_boundaries(self.current_raster_boundary_data, self.output_file_path) # PNG needs the raster data
            else:
                raise ValueError(f"Unsupported output format: {self.output_file_format}")
            self.status_label.setText(f"File saved successfully to {self.output_file_path}")
        except Exception as e:
            self.status_label.setText(f"Error saving file: {e}")

    def _save_file_as(self):
        if self.current_raster_boundary_data is None: # Check for the actual boundary image data
            return # Should not happen if action is disabled

        # Offer both SVG and PNG options
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Boundaries As...", "", 
            "SVG Files (*.svg);;PNG Image (*.png)"
        )
        if file_path: # If a path was selected
            self.output_file_path = file_path
            if selected_filter == "SVG Files (*.svg)":
                self.output_file_format = 'svg'
            elif selected_filter == "PNG Image (*.png)":
                self.output_file_format = 'png'
            else: # Fallback, though should be covered by filters
                self.output_file_format = 'svg' 
            self._save_file()

    def start_processing(self):
        if self.input_img_data is None:
            self.status_label.setText("Please select an input image first.")
            return

        self.status_label.setText("Processing...")

        num_colours = self.num_colours_slider.value() # Get value from slider

        # Setup thread and worker
        self.thread = QThread()
        self.worker = Worker(self.input_img_data, num_colours) # No smoothing_level passed
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_processing_finished(self, quantised_label_grid): # Only receive raw_label_grid
        self.quantised_label_grid = quantised_label_grid # Store it
        self.status_label.setText("Processing complete. Preview generated. Ready to save.")

        # Now, apply smoothing and update preview based on current slider value
        self._update_smoothed_preview(self.smoothing_slider.value())

        self.save_action.setEnabled(True)
        self.save_as_action.setEnabled(True)

    def _update_smoothed_preview(self, smoothing_level: int):
        # Scale the integer slider value to a float smoothing level (e.g., 0-100 -> 0.0-10.0)
        float_smoothing_level = smoothing_level / 10.0
        self.smoothing_label.setText(f"{float_smoothing_level:.1f}") # Update label for slider with float

        if self.quantised_label_grid is None:
            return # No image loaded or processed yet, but allow slider value to be set

        if float_smoothing_level > 0:
            # Calculate an odd kernel size based on the float smoothing level
            # Ensure it's at least 1 and always odd.
            kernel_size = max(1, int(2 * float_smoothing_level + 1))
            if kernel_size % 2 == 0: # Ensure kernel size is odd
                kernel_size += 1
            self.smoothed_label_grid = median_filter(self.quantised_label_grid, size=kernel_size)
        else:
            self.smoothed_label_grid = self.quantised_label_grid # No smoothing, use raw grid

        boundary_preview_img = create_raster_boundaries(self.smoothed_label_grid)
        boundary_pixmap = self.numpy_to_qpixmap(boundary_preview_img)
        self.img_display_out.setPixmap(boundary_pixmap)
        self.current_raster_boundary_data = boundary_preview_img # Store for PNG saving

    @staticmethod
    def numpy_to_qpixmap(np_img: np.ndarray) -> QPixmap:
        """Converts a NumPy array to a QPixmap."""
        if np_img.ndim == 3:  # RGB
            height, width, channel = np_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(np_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:  # Greyscale
            height, width = np_img.shape
            bytes_per_line = width
            q_img = QImage(np_img.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(q_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = BoundaryDetectorApp()
    ex.show()
    sys.exit(app.exec())