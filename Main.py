############################################ ADHAM'S GUI USING TKINTER INSTEAD OF PYQT ##########################################
# import cv2
# import numpy as np
# import time
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk

# # Function to load an image
# def load_image():
#     global img, img_gray, img_display

#     file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
#     if not file_path:
#         return

#     # Read and process image
#     img = cv2.imread(file_path)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Convert to display format
#     img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_display = ImageTk.PhotoImage(Image.fromarray(cv2.resize(img_display, (300, 300))))

#     # Show image in left panel
#     label_input.config(image=img_display)
#     label_input.image = img_display

# # Function to apply Harris Corner Detector
# def compute_gradients(img_gray):
#     """Compute image gradients using Sobel filters."""
#     I_x, I_y, magn = sobel_detection(img_gray)

#     return I_x, I_y

# def sobel_detection(image):


#     sobel_x = np.array([[-1, 0, 1],
#                         [-2, 0, 2],
#                         [-1, 0, 1]])

#     sobel_y = np.array([[-1, -2, -1],
#                         [0, 0, 0],
#                         [1, 2, 1]])

#     edge_x = convolve_2d(image, sobel_x)
#     edge_y = convolve_2d(image, sobel_y)

#     edge_magnitude = np.hypot(edge_x, edge_y)
#     edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255

#     return edge_x.astype(np.uint8), edge_y.astype(np.uint8), edge_magnitude.astype(np.uint8)


# def convolve_2d(image, kernel):

#     image_height, image_width = image.shape
#     kernel_height, kernel_width = kernel.shape

#     # Compute padding size
#     pad_h = kernel_height // 2
#     pad_w = kernel_width // 2

#     # Pad image to keep original size after filtering
#     padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

#     output = np.zeros((image_height, image_width), dtype=np.float32)

#     # Perform convolution
#     for i in range(image_height):
#         for j in range(image_width):
#             region = padded_image[i:i + kernel_height, j:j + kernel_width]

#             # Apply element-wise multiplication and sum the result
#             output[i, j] = np.sum(region * kernel)

#     # Normalize and return the output image
#     return np.clip(output, 0, 255).astype(np.uint8)




# def compute_harris_response(I_x, I_y, k=0.04):
#     """Manually compute the Harris Corner response map."""
#     # Compute second moment matrix components
#     I_x2 = I_x ** 2
#     I_y2 = I_y ** 2
#     I_xy = I_x * I_y

#     # Apply Gaussian Blur to smooth out noise
#     I_x2 = cv2.GaussianBlur(I_x2, (3, 3), 1)
#     I_y2 = cv2.GaussianBlur(I_y2, (3, 3), 1)
#     I_xy = cv2.GaussianBlur(I_xy, (3, 3), 1)

#     # Compute determinant and trace of M
#     det_M = (I_x2 * I_y2) - (I_xy ** 2)
#     trace_M = I_x2 + I_y2

#     # Compute Harris response
#     R = det_M - k * (trace_M ** 2)

#     return R


# def mark_corners(img, harris_response, threshold_ratio=0.95):
#     """Mark detected corners on the image."""
#     img_marked = img.copy()
#     threshold = threshold_ratio * harris_response.max()
#     img_marked[harris_response > threshold] = [0, 0, 255]  # Mark corners in red
#     return img_marked


# def apply_harris_corner_detection():
#     """Full pipeline for Harris Corner Detection."""
#     global img, img_gray, img_display

#     if img is None:
#         return

#     I_x, I_y = compute_gradients(img_gray)
#     harris_response = compute_harris_response(I_x, I_y)
#     img_corners = mark_corners(img, harris_response)

#     # Resize result image to 250x250
#     img_corners_resized = cv2.resize(img_corners, (250, 250))

#     # Convert to display format
#     img_corners_display = cv2.cvtColor(img_corners_resized, cv2.COLOR_BGR2RGB)
#     img_corners_display = ImageTk.PhotoImage(Image.fromarray(img_corners_display))

#     # Show image in right panel
#     label_output.config(image=img_corners_display)
#     label_output.image = img_corners_display
# # # GUI setup
# # root = tk.Tk()
# # root.title("Harris Corner Detector")
# # root.geometry("650x450")
# # root.configure(bg="#f0f0f0")

# # # Title Label
# # title_label = tk.Label(root, text="Harris Corner Detector", font=("Arial", 16, "bold"), bg="#f0f0f0")
# # title_label.pack(pady=10)

# # # Frame for Buttons
# # button_frame = tk.Frame(root, bg="#f0f0f0")
# # button_frame.pack()

# # btn_load = tk.Button(button_frame, text="Load Image", command=load_image, font=("Arial", 12), bg="#007bff", fg="white")
# # btn_load.grid(row=0, column=0, padx=10, pady=5)

# # btn_harris = tk.Button(button_frame, text="Apply Harris Detector", command=apply_harris_corner_detection, font=("Arial", 12), bg="#28a745", fg="white")
# # btn_harris.grid(row=0, column=1, padx=10, pady=5)

# # time_label = tk.Label(root, text="Computation Time: -- sec", font=("Arial", 12), bg="#f0f0f0")
# # time_label.pack(pady=5)

# # # Frame for Image Displays
# # image_frame = tk.Frame(root, bg="#f0f0f0")
# # image_frame.pack()

# # # Placeholder Image
# # placeholder = ImageTk.PhotoImage(Image.new("RGB", (300, 300), (200, 200, 200)))

# # label_input = tk.Label(image_frame, image=placeholder, bg="white", width=300, height=300)
# # label_input.grid(row=0, column=0, padx=10, pady=10)
# # label_input.image = placeholder
# # label_input_text = tk.Label(image_frame, text="Original Image", font=("Arial", 12), bg="#f0f0f0")
# # label_input_text.grid(row=1, column=0, pady=5)

# # label_output = tk.Label(image_frame, image=placeholder, bg="white", width=300, height=300)
# # label_output.grid(row=0, column=1, padx=10, pady=10)
# # label_output.image = placeholder
# # label_output_text = tk.Label(image_frame, text="Processed Image", font=("Arial", 12), bg="#f0f0f0")
# # label_output_text.grid(row=1, column=1, pady=5)

# # # Run GUI
# # root.mainloop()

import sys
from PyQt5.QtWidgets import QApplication,QMessageBox,QGraphicsLineItem, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTabWidget, QSlider, QGraphicsView, QGraphicsScene, QTextEdit,QFileDialog,QGraphicsPixmapItem
from PyQt5.QtCore import Qt,QThread, pyqtSignal 
from PyQt5.QtGui import QPixmap,QImage,QPainter, QPen
import  cv2
from PIL import Image,ImageDraw
import numpy as np
import time 
from scipy.spatial import distance , cKDTree

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
        self.lambda_button.clicked.connect(self.apply_lambda)
        
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
    
    

    def apply_lambda(self):
        pass
    
    def apply_sift(self):
        pass

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

        # Apply Gaussian Blur to smooth out noise
        I_x2 = cv2.GaussianBlur(I_x2, (3, 3), 1)
        I_y2 = cv2.GaussianBlur(I_y2, (3, 3), 1)
        I_xy = cv2.GaussianBlur(I_xy, (3, 3), 1)

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
        ###################### WAITING FOR FUNCTION SIFT TO TEST SSD AND NCC ########################
        ####################### BECAUSE NEED KETPOINTS AND DESCRIPTORS OUTPUT FROM SIFT ###########################

    def apply_ssd(self):
        """Compute SSD between loaded images."""
        if not hasattr(self, 'image1') or not hasattr(self, 'image2'):
            QMessageBox.warning(self, "Error", "Please load both images first!")
            return

        start_time = time.time()

        # Resize images to match
        self.img1_resized, self.img2_resized = self.resize_images_to_match(self.image1, self.image2)

        ssd = np.sum((self.img1_resized.astype(np.float64) - self.img2_resized.astype(np.float64)) ** 2)
        end_time = time.time()
        self.computation_time.setText(f"{(end_time - start_time) * 1000:.2f} ms")
        
        QMessageBox.information(self, "SSD Result", f"SSD Value: {ssd}")
        # Display matched features
        # self.display_matched_features(self.img1_resized, self.img2_resized)
        self.process_images("SSD")
    
    def apply_ncc(self):
        """Compute NCC between loaded images."""
        if not hasattr(self, 'image1') or not hasattr(self, 'image2'):
            QMessageBox.warning(self, "Error", "Please load both images first!")
            return

        start_time = time.time()

        # Resize images to match
        self.img1_resized, self.img2_resized = self.resize_images_to_match(self.image1, self.image2)

        # Compute NCC
        mean1 = np.mean(self.img1_resized)
        mean2 = np.mean(self.img2_resized)
        numerator = np.sum((self.img1_resized - mean1) * (self.img2_resized - mean2))
        denominator = np.sqrt(np.sum((self.img1_resized - mean1) ** 2) * np.sum((self.img2_resized - mean2) ** 2))
        ncc = numerator / (denominator + 1e-8)  

        end_time = time.time()
        self.computation_time.setText(f"{(end_time - start_time) * 1000:.2f} ms")

        QMessageBox.information(self, "NCC Result", f"NCC Value: {ncc:.4f}")
        # Display matched features
        #self.display_matched_features(self.img1_resized, self.img2_resized)
        self.process_images("NCC")

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

        # Draw matching feature lines
        pen = QPen(Qt.red)
        pen.setWidth(2)
        for match in matches:
            idx1 = match[0].queryIdx
            idx2 = match[0].trainIdx
            x1, y1 = keypoints1[idx1].pt
            x2, y2 = keypoints2[idx2].pt
            x2 += width1  # Adjust x-coordinate for the second image

            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(pen)
            scene.addItem(line)

        output_view.setScene(scene)

    def process_images(self, method):
        """Process images and draw feature matches using SSD or NCC."""
        if not hasattr(self, 'image1') or not hasattr(self, 'image2'):
            QMessageBox.warning(self, "Error", "Please load both images first!")
            return
        
        # Convert images to grayscale
        image1_gray = cv2.cvtColor(self.image1, cv2.COLOR_RGB2GRAY)
        image2_gray = cv2.cvtColor(self.image2, cv2.COLOR_RGB2GRAY)

        # Extract keypoints using SIFT (assuming apply_sift exists)
        keypoints1, descriptors1 = FeatureExtractionTab.apply_sift(image1_gray)
        keypoints2, descriptors2 = FeatureExtractionTab.apply_sift(image2_gray)
        
        if descriptors1 is None or descriptors2 is None:
            QMessageBox.warning(self, "Error", "Feature extraction failed!")
            return
        
        # Perform feature matching
        matches = []
        for i, desc1 in enumerate(descriptors1):
            best_match = None
            best_score = float('inf') if method == "SSD" else -float('inf')
            best_idx = -1
            
            for j, desc2 in enumerate(descriptors2):
                if method == "SSD":
                    score = np.sum((desc1 - desc2) ** 2)
                    if score < best_score:
                        best_score = score
                        best_match = (i, j)
                else:  # NCC
                    mean1, mean2 = np.mean(desc1), np.mean(desc2)
                    num = np.sum((desc1 - mean1) * (desc2 - mean2))
                    den = np.sqrt(np.sum((desc1 - mean1) ** 2) * np.sum((desc2 - mean2) ** 2))
                    score = num / (den + 1e-8)
                    if score > best_score:
                        best_score = score
                        best_match = (i, j)
            
            if best_match:
                matches.append([cv2.DMatch(best_match[0], best_match[1], best_score)])
        
        # Draw feature matches in QGraphicsView
        self.draw_feature_matches(self.output_view, self.image1, self.image2, keypoints1, keypoints2, matches)

    # def display_matched_features(self, img1, img2):
    #     """Display matched features between two images with lines connecting points."""
    #     # orb = cv2.ORB_create()
    #     # kp1, des1 = orb.detectAndCompute(img1, None)
    #     # kp2, des2 = orb.detectAndCompute(img2, None)

    #     # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #     # matches = bf.match(des1, des2)
    #     # matches = sorted(matches, key=lambda x: x.distance)

    #     # matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     # self.display_image(matched_img, self.output_view)
        

    # def display_image(self, img, view):
    #     """Convert and display an image in QGraphicsView."""
    #     height, width, channel = img.shape
    #     bytes_per_line = 3 * width
    #     q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    #     q_pixmap = QPixmap.fromImage(q_image)
    #     scene = QGraphicsScene()
    #     scene.addItem(QGraphicsPixmapItem(q_pixmap))
    #     view.setScene(scene)


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
