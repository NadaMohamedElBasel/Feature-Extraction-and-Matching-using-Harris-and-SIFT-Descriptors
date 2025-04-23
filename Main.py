

import sys
from PyQt5.QtWidgets import QApplication,QMessageBox,QGraphicsLineItem, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTabWidget, QSlider, QGraphicsView, QGraphicsScene, QTextEdit,QFileDialog,QGraphicsPixmapItem
from PyQt5.QtCore import Qt,QThread, pyqtSignal 
from PyQt5.QtGui import QPixmap,QImage,QPainter, QPen , QColor
import  cv2
from PIL import Image,ImageDraw
import numpy as np
import time 
from scipy.spatial import distance , cKDTree
from scipy.ndimage import maximum_filter

class FeatureExtractionTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # Image Viewports
        self.input_label = QLabel("Original Image")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_view = QGraphicsView()
        self.output_label = QLabel("Processed Image")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_view = QGraphicsView()
        # Load Image Button
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        # Buttons
        self.harris_button = QPushButton("Harris")
        self.harris_button.clicked.connect(self.apply_harris_corner_detection)
        self.lambda_button = QPushButton("λ-")
        self.lambda_button.clicked.connect(self.apply_lambda)
        self.sift_button = QPushButton("SIFT")
        self.sift_button.clicked.connect(self.apply_sift)
        
        # Slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(7)
        self.threshold_slider.setValue(3)
        self.threshold_value = QLabel("3")
        self.threshold_slider.valueChanged.connect(lambda: self.threshold_value.setText(str(self.threshold_slider.value())))
        
        # Computation Time Label
        self.computation_label = QLabel("Computation time: ")
        self.computation_time = QLabel("0 ms")
        
        # Layouts
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_view)
        
        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_view)
        
        img_layout = QHBoxLayout()
        img_layout.addLayout(input_layout)
        img_layout.addLayout(output_layout)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.harris_button)
        btn_layout.addWidget(self.lambda_button)
        btn_layout.addWidget(self.sift_button)
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Threshold:"))
        slider_layout.addWidget(self.threshold_slider)
        slider_layout.addWidget(self.threshold_value)
        
        comp_layout = QHBoxLayout()
        comp_layout.addWidget(self.computation_label)
        comp_layout.addWidget(self.computation_time)
        comp_layout.addWidget(self.load_image_button)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(img_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(slider_layout)
        main_layout.addLayout(comp_layout)
        
        self.setLayout(main_layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)")
        if not file_path:
            return

        # Read and process image
        img = cv2.imread(file_path)
        self.img=img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display

        # Convert OpenCV image (NumPy array) to QImage
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        q_pixmap = QPixmap.fromImage(q_image)

        # Display image in QGraphicsView
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.input_view.setScene(scene)
    
    
    def make_gaussian_mask(self, size, sigma=1):
        """Return a 2D Gaussian kernel."""
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        return kernel

    def apply_lambda(self): 
        if not hasattr(self, 'img') or self.img is None:
            print("No image loaded. Please load an image first.")
            return

        start_time = time.time()

        # Compute gradients
        I_x, I_y = self.compute_gradients(self.img_gray)

        # Create Gaussian kernel
        kernel_size = 3
        gaussian_kernel = self.make_gaussian_mask(kernel_size).reshape((kernel_size, kernel_size))

        # Apply custom convolution with Gaussian kernel
        I_x2 = self.convolve_2d((I_x.astype(np.float32) ** 2), gaussian_kernel)
        I_y2 = self.convolve_2d((I_y.astype(np.float32) ** 2), gaussian_kernel)
        I_xy = self.convolve_2d((I_x.astype(np.float32) * I_y.astype(np.float32)), gaussian_kernel)

        # Determinant and trace
        det_M = I_x2 * I_y2 - I_xy ** 2
        trace_M = I_x2 + I_y2

        # Eigenvalues
        temp = np.sqrt(np.maximum(trace_M ** 2 - 4 * det_M, 0))
        lambda1 = (trace_M + temp) / 2
        lambda2 = (trace_M - temp) / 2
        lam_min = np.minimum(lambda1, lambda2)

        # Threshold and mark
        threshold_ratio = float(self.threshold_slider.value()) / 7.0
        threshold = threshold_ratio * lam_min.max()
        img_marked = self.img.copy()
        img_marked[lam_min > threshold] = [0, 255, 0]  # Mark λ⁻ points in green

        # Resize for display
        img_out = cv2.resize(img_marked, (250, 250))
        elapsed = (time.time() - start_time) * 1000
        self.computation_time.setText(f"{elapsed:.2f} ms")

        # Convert to QImage and display
        height, width, channel = img_out.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_out.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene)

    def apply_sift(self):
        """Apply SIFT (Scale-Invariant Feature Transform) for feature extraction."""
        if not hasattr(self, 'img') or self.img is None:
            QMessageBox.warning(self, "Error", "Please load an image first!")
            return

        start_time = time.time()

        # Convert image to grayscale 
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Harris Corner Detection for Keypoints
        dst = self.apply_harris_corner_detection() # changed instead of cv2.cornerharris
        dst = cv2.dilate(dst, None)
        keypoints = np.argwhere(dst > 0.99 * dst.max())  # Using a threshold to detect points

        # Sobel detection 
        edge_x, edge_y, edge_magnitude = self.sobel_detection(img_gray)

        # Ensure descriptor size is fixed by defining a patch size
        patch_size = 16  
        descriptors = []

        for kp in keypoints:
            y, x = kp
            # Create a small patch around the keypoint ensuring the patch is within image bounds
            patch = img_gray[max(0, y - patch_size // 2):min(img_gray.shape[0], y + patch_size // 2),
                            max(0, x - patch_size // 2):min(img_gray.shape[1], x + patch_size // 2)]

            # Ensure the patch is exactly of the desired patch_size
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                # Compute gradients (edge_x, edge_y, edge_magnitude) for this patch
                grad_x_patch = edge_x[max(0, y - patch_size // 2):min(edge_x.shape[0], y + patch_size // 2),
                                    max(0, x - patch_size // 2):min(edge_x.shape[1], x + patch_size // 2)]
                grad_y_patch = edge_y[max(0, y - patch_size // 2):min(edge_y.shape[0], y + patch_size // 2),
                                    max(0, x - patch_size // 2):min(edge_y.shape[1], x + patch_size // 2)]
                magnitude_patch = edge_magnitude[max(0, y - patch_size // 2):min(edge_magnitude.shape[0], y + patch_size // 2),
                                                max(0, x - patch_size // 2):min(edge_magnitude.shape[1], x + patch_size // 2)]

                # Flatten gradients and magnitudes and concatenate them to form a descriptor vector
                descriptor = np.concatenate([grad_x_patch.flatten(), grad_y_patch.flatten(), magnitude_patch.flatten()])
                descriptors.append(descriptor)

        # Ensure descriptors are all the same length and convert to a numpy array
        descriptors = np.array(descriptors)
        if descriptors.shape[1] != patch_size * patch_size * 3:  # Checking if the descriptor is valid
            QMessageBox.warning(self, "Error", "Descriptors have inconsistent shapes!")


        # Draw keypoints on the image (for visualization)
        img_with_keypoints = cv2.drawKeypoints(self.img, [cv2.KeyPoint(float(x), float(y), 1) for y, x in keypoints], None)
        img_with_keypoints = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display


        # Calculate computation time
        end_time = time.time()
        self.computation_time.setText(f"{(end_time - start_time) * 1000:.2f} ms")

        # Convert to QImage format for PyQt
        height, width, channel = img_with_keypoints.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_with_keypoints.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display in QGraphicsView
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene)

        
        self.keypoints = keypoints
        self.descriptors = descriptors

        # Return keypoints and descriptors
        return keypoints, descriptors

    # Function to apply Harris Corner Detector
    def compute_gradients(self,img_gray):
        """Compute image gradients using Sobel filters."""
        I_x, I_y, magn = self.sobel_detection(img_gray)

        return I_x, I_y

    def sobel_detection(self,image):


        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        edge_x = self.convolve_2d(image, sobel_x)
        edge_y = self.convolve_2d(image, sobel_y)

        edge_magnitude = np.hypot(edge_x, edge_y)
        edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255

        return edge_x.astype(np.uint8), edge_y.astype(np.uint8), edge_magnitude.astype(np.uint8)


    def convolve_2d(self,image, kernel):

        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # Compute padding size
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2

        # Pad image to keep original size after filtering
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        output = np.zeros((image_height, image_width), dtype=np.float32)

        # Perform convolution
        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]

                # Apply element-wise multiplication and sum the result
                output[i, j] = np.sum(region * kernel)

        # Normalize and return the output image
        return np.clip(output, 0, 255).astype(np.uint8)

    def compute_harris_response(self,I_x, I_y, k=0.04):
        """Manually compute the Harris Corner response map."""
        # Compute second moment matrix components
        I_x2 = I_x ** 2
        I_y2 = I_y ** 2
        I_xy = I_x * I_y

        # Create Gaussian kernel
        gaussian_kernel = self.make_gaussian_mask(size=3, sigma=1)

        # Manually apply Gaussian filter using convolution
        I_x2 = self.convolve_2d(I_x2, gaussian_kernel)
        I_y2 = self.convolve_2d(I_y2, gaussian_kernel)
        I_xy = self.convolve_2d(I_xy, gaussian_kernel)

        # Compute determinant and trace of M
        det_M = (I_x2 * I_y2) - (I_xy ** 2)
        trace_M = I_x2 + I_y2

        # Compute Harris response
        R = det_M - k * (trace_M ** 2)

        return R

    def mark_corners(self,img, harris_response, threshold_ratio):
        """Mark detected corners on the image."""
        img_marked = img.copy()
        threshold = threshold_ratio * harris_response.max()
        img_marked[harris_response > threshold] = [0, 0, 255]  # Mark corners in red
        return img_marked


    def apply_harris_corner_detection(self):
        """Full pipeline for Harris Corner Detection using PyQt, without global variables."""
        if not hasattr(self, 'img') or self.img is None:
            print("No image loaded. Please load an image first.")
            return
        start_time = time.time()

        # Get threshold value from slider
        threshold_ratio = float(self.threshold_slider.value()) / 7.0 
        # Compute gradients and Harris response
        I_x, I_y = self.compute_gradients(self.img_gray)
        harris_response = self.compute_harris_response(I_x, I_y)
        img_corners = self.mark_corners(self.img, harris_response,threshold_ratio)

        # Resize result image to 250x250
        img_corners_resized = cv2.resize(img_corners, (250, 250))
        end_time = time.time()
        self.computation_time.setText(f"{(end_time - start_time) * 1000:.2f} ms")
        # Convert to QImage format for PyQt
        height, width, channel = img_corners_resized.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_corners_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display in QGraphicsView
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene) 
        return harris_response 

class FeatureMatchingTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # Image Viewports
        self.image1_label = QLabel("First Image")
        self.image1_label.setAlignment(Qt.AlignCenter)
        self.image1_view = QGraphicsView()
        self.image2_label = QLabel("Second Image")
        self.image2_label.setAlignment(Qt.AlignCenter)
        self.image2_view = QGraphicsView()
        self.output_label = QLabel("Output Image")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_view = QGraphicsView()
        # Load Image Buttons
        self.load_first_image_button = QPushButton("Load First Image")
        self.load_first_image_button.clicked.connect(self.load_first_image)
        self.load_second_image_button = QPushButton("Load Second Image")
        self.load_second_image_button.clicked.connect(self.load_second_image)
        # Buttons
        self.ssd_button = QPushButton("SSD")
        self.ssd_button.clicked.connect(self.apply_ssd)
        self.ncc_button = QPushButton("NCC")
        self.ncc_button.clicked.connect(self.apply_ncc)
        
        # Computation Time Label
        self.computation_label = QLabel("Computation time: ")
        self.computation_time = QLabel("0 ms")
        
        # Layouts
        first_img_layout = QVBoxLayout()
        first_img_layout.addWidget(self.image1_label)
        first_img_layout.addWidget(self.image1_view)
        first_img_layout.addWidget(self.load_first_image_button)
        
        second_img_layout = QVBoxLayout()
        second_img_layout.addWidget(self.image2_label)
        second_img_layout.addWidget(self.image2_view)
        second_img_layout.addWidget(self.load_second_image_button)
        
        output_img_layout = QVBoxLayout()
        output_img_layout.addWidget(self.output_label)
        output_img_layout.addWidget(self.output_view)
        
        img_layout = QHBoxLayout()
        img_layout.addLayout(first_img_layout)
        img_layout.addLayout(second_img_layout)
        img_layout.addLayout(output_img_layout)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.ssd_button)
        btn_layout.addWidget(self.ncc_button)
        
        comp_layout = QHBoxLayout()
        comp_layout.addWidget(self.computation_label)
        comp_layout.addWidget(self.computation_time)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(img_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(comp_layout)
        
        self.setLayout(main_layout)

    def load_first_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)")
        if not file_path:
            return

        # Read and process image
        self.image1 = cv2.imread(file_path)  # Store in self.image1
        self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display

        # Convert OpenCV image (NumPy array) to QImage
        height, width, channel = self.image1.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.image1.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        q_pixmap = QPixmap.fromImage(q_image)

        # Display image in QGraphicsView
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.image1_view.setScene(scene)

    def load_second_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)")
        if not file_path:
            return

        # Read and process image
        self.image2 = cv2.imread(file_path)  # Store in self.image2
        self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display

        # Convert OpenCV image (NumPy array) to QImage
        height, width, channel = self.image2.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.image2.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        q_pixmap = QPixmap.fromImage(q_image)

        # Display image in QGraphicsView
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.image2_view.setScene(scene)
    
    def resize_images_to_match(self, img1, img2):
        """Resize images to the same dimensions for comparison."""
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        return img1_resized, img2_resized
    
    def apply_sift(self,img):
        """Apply SIFT (Scale-Invariant Feature Transform) for feature extraction."""
        start_time = time.time()
        # Convert image to grayscale 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        # Harris Corner Detection for Keypoints
        dst = self.apply_harris_corner_detection(img_gray) # changed instead of cv2.cornerharris
        dst = cv2.dilate(dst,kernel)
        keypoints = np.argwhere(dst > 0.99 * dst.max())  # Using a threshold to detect points

        # Sobel detection 
        edge_x, edge_y, edge_magnitude = self.sobel_detection(img_gray)

        # Ensure descriptor size is fixed by defining a patch size
        patch_size = 16  
        descriptors = []

        for kp in keypoints:
            y, x = kp
            # Create a small patch around the keypoint ensuring the patch is within image bounds
            patch = img_gray[max(0, y - patch_size // 2):min(img_gray.shape[0], y + patch_size // 2),
                            max(0, x - patch_size // 2):min(img_gray.shape[1], x + patch_size // 2)]

            # Ensure the patch is exactly of the desired patch_size
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                # Compute gradients (edge_x, edge_y, edge_magnitude) for this patch
                grad_x_patch = edge_x[max(0, y - patch_size // 2):min(edge_x.shape[0], y + patch_size // 2),
                                    max(0, x - patch_size // 2):min(edge_x.shape[1], x + patch_size // 2)]
                grad_y_patch = edge_y[max(0, y - patch_size // 2):min(edge_y.shape[0], y + patch_size // 2),
                                    max(0, x - patch_size // 2):min(edge_y.shape[1], x + patch_size // 2)]
                magnitude_patch = edge_magnitude[max(0, y - patch_size // 2):min(edge_magnitude.shape[0], y + patch_size // 2),
                                                max(0, x - patch_size // 2):min(edge_magnitude.shape[1], x + patch_size // 2)]

                # Flatten gradients and magnitudes and concatenate them to form a descriptor vector
                descriptor = np.concatenate([grad_x_patch.flatten(), grad_y_patch.flatten(), magnitude_patch.flatten()])
                descriptors.append(descriptor)

        # Ensure descriptors are all the same length and convert to a numpy array
        descriptors = np.array(descriptors)
        if descriptors.shape[1] != patch_size * patch_size * 3:  # Checking if the descriptor is valid
            QMessageBox.warning(self, "Error", "Descriptors have inconsistent shapes!")


        # Draw keypoints on the image (for visualization)
        img_with_keypoints = cv2.drawKeypoints(img, [cv2.KeyPoint(float(x), float(y), 1) for y, x in keypoints], None)
        img_with_keypoints = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display


        # Calculate computation time
        end_time = time.time()
        self.computation_time.setText(f"{(end_time - start_time) * 1000:.2f} ms")

        # Convert to QImage format for PyQt
        height, width, channel = img_with_keypoints.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_with_keypoints.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display in QGraphicsView
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene)

        self.keypoints = keypoints
        self.descriptors = descriptors

        # Return keypoints and descriptors
        return keypoints, descriptors

    def match_features(self, desc1, desc2, method="SSD"):
        matches = []
        for i, d1 in enumerate(desc1):
            best_idx = -1
            best_score = float('inf') if method == "SSD" else -float('inf')

            for j, d2 in enumerate(desc2):
                if method == "SSD":
                    score = np.sum((d1 - d2) ** 2)
                    if score < best_score:
                        best_score, best_idx = score, j
                else:  # NCC
                    mean1, mean2 = np.mean(d1), np.mean(d2)
                    numerator = np.sum((d1 - mean1) * (d2 - mean2))
                    denominator = np.sqrt(np.sum((d1 - mean1) ** 2) * np.sum((d2 - mean2) ** 2))
                    score = numerator / (denominator + 1e-8)
                    if score > best_score:
                        best_score, best_idx = score, j

            if best_idx != -1:
                matches.append([cv2.DMatch(_queryIdx=i, _trainIdx=best_idx, _distance=float(best_score))])
        return matches
    
    def apply_comparison(self, method="SSD"):
        if not hasattr(self, 'image1') or not hasattr(self, 'image2'):
            QMessageBox.warning(self, "Error", "Please load both images first!")
            return

        start_time = time.time()
        self.img1_resized, self.img2_resized = self.resize_images_to_match(self.image1, self.image2)

        keypoints1, desc1 = self.apply_sift(self.img1_resized)
        keypoints2, desc2 = self.apply_sift(self.img2_resized)

        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            QMessageBox.warning(self, "Error", "Feature extraction failed!")
            return

        matches = self.match_features(desc1, desc2, method)

        end_time = time.time()
        self.computation_time.setText(f"{(end_time - start_time) * 1000:.2f} ms")

        if method == "SSD":
            value = sum(m[0].distance for m in matches)
            QMessageBox.information(self, "SSD Result", f"SSD Value: {value:.2f}")
        else:
            value = sum(m[0].distance for m in matches) / len(matches)
            QMessageBox.information(self, "NCC Result", f"NCC Value: {value:.4f}")

        self.draw_feature_matches(self.output_view, self.image1, self.image2, keypoints1, keypoints2, matches)

    def draw_feature_matches(self,output_view, image1, image2, keypoints1, keypoints2, matches):
        """Draw feature matches manually in QGraphicsView."""
        # Convert NumPy images to QImage
        height1, width1, _ = self.image1.shape
        height2, width2, _ = self.image2.shape
        combined_width = width1 + width2  # Combine images side by side
        combined_height = max(height1, height2)

        # Create a blank canvas
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_image[:height1, :width1] = self.image1
        combined_image[:height2, width1:width1 + width2] = self.image2

        # Convert OpenCV image to QImage
        bytes_per_line = 3 * combined_width
        q_image = QImage(combined_image.data, combined_width, combined_height, bytes_per_line, QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)

        # Display image in QGraphicsView
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        # Sort matches by distance
        matches = sorted(matches, key=lambda match: match[0].distance)
        # Draw matching feature lines
        pen = QPen(Qt.red)
        pen.setWidth(2)
        for match in matches[:30]:
            idx1 = match[0].queryIdx
            idx2 = match[0].trainIdx
            y1, x1 = keypoints1[idx1]  # Remember, keypoints are (row, col) → (y, x)
            y2, x2 = keypoints2[idx2]

            # Swap to (x, y) format for display
            x1, y1 = float(x1), float(y1)
            x2, y2 = float(x2), float(y2)
            x2 += width1  # Adjust x-coordinate for the second image

            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(pen)
            scene.addItem(line)

        output_view.setScene(scene)
    
    def apply_ssd(self):
        self.apply_comparison(method="SSD")

    def apply_ncc(self):
        self.apply_comparison(method="NCC")

    def make_gaussian_mask(self, size, sigma=1):
            """Return a 2D Gaussian kernel."""
            ax = np.linspace(-(size // 2), size // 2, size)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel /= np.sum(kernel)
            return kernel 

# Function to apply Harris Corner Detector
    def compute_gradients(self,img_gray):
        """Compute image gradients using Sobel filters."""
        I_x, I_y, magn = self.sobel_detection(img_gray)

        return I_x, I_y

    def sobel_detection(self,image):


        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        edge_x = self.convolve_2d(image, sobel_x)
        edge_y = self.convolve_2d(image, sobel_y)

        edge_magnitude = np.hypot(edge_x, edge_y)
        edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255

        return edge_x.astype(np.uint8), edge_y.astype(np.uint8), edge_magnitude.astype(np.uint8)


    def convolve_2d(self,image, kernel):

        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # Compute padding size
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2

        # Pad image to keep original size after filtering
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        output = np.zeros((image_height, image_width), dtype=np.float32)

        # Perform convolution
        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]

                # Apply element-wise multiplication and sum the result
                output[i, j] = np.sum(region * kernel)

        # Normalize and return the output image
        return np.clip(output, 0, 255).astype(np.uint8)

    def compute_harris_response(self,I_x, I_y, k=0.04):
        """Manually compute the Harris Corner response map."""
        # Compute second moment matrix components
        I_x2 = I_x ** 2
        I_y2 = I_y ** 2
        I_xy = I_x * I_y

        # Create Gaussian kernel
        gaussian_kernel = self.make_gaussian_mask(size=3, sigma=1)

        # Manually apply Gaussian filter using convolution
        I_x2 = self.convolve_2d(I_x2, gaussian_kernel)
        I_y2 = self.convolve_2d(I_y2, gaussian_kernel)
        I_xy = self.convolve_2d(I_xy, gaussian_kernel)

        # Compute determinant and trace of M
        det_M = (I_x2 * I_y2) - (I_xy ** 2)
        trace_M = I_x2 + I_y2

        # Compute Harris response
        R = det_M - k * (trace_M ** 2)

        return R

    def mark_corners(self, img, harris_response, threshold_ratio):
        """Mark detected corners on the image."""
        # Convert grayscale image to BGR if needed
        if len(img.shape) == 2 or img.shape[2] == 1:
            img_marked = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_marked = img.copy()

        threshold = threshold_ratio * harris_response.max()
        img_marked[harris_response > threshold] = [0, 0, 255]  # Mark corners in red
        return img_marked

    def apply_harris_corner_detection(self,img_gray):
        """Full pipeline for Harris Corner Detection using PyQt, without global variables."""
        start_time = time.time()
        # Compute gradients and Harris response
        I_x, I_y = self.compute_gradients(img_gray)
        harris_response = self.compute_harris_response(I_x, I_y)
        img_corners = self.mark_corners(img_gray, harris_response,6.0/7)
        # Resize result image to 250x250
        img_corners_resized = cv2.resize(img_corners, (250, 250))
        end_time = time.time()
        self.computation_time.setText(f"{(end_time - start_time) * 1000:.2f} ms")
        # Convert to QImage format for PyQt
        height, width, channel = img_corners_resized.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_corners_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display in QGraphicsView
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene) 
        return harris_response 


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Extraction & Matching")
        self.setGeometry(100, 100, 800, 600)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(FeatureExtractionTab(), "Feature Extraction")
        self.tabs.addTab(FeatureMatchingTab(), "Feature Matching")
        
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
