import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Function to load an image
def load_image():
    global img, img_gray, img_display

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return

    # Read and process image
    img = cv2.imread(file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to display format
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = ImageTk.PhotoImage(Image.fromarray(cv2.resize(img_display, (300, 300))))

    # Show image in left panel
    label_input.config(image=img_display)
    label_input.image = img_display

# Function to apply Harris Corner Detector
def compute_gradients(img_gray):
    """Compute image gradients using Sobel filters."""
    I_x, I_y, magn = sobel_detection(img_gray)

    return I_x, I_y

def sobel_detection(image):


    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    edge_x = convolve_2d(image, sobel_x)
    edge_y = convolve_2d(image, sobel_y)

    edge_magnitude = np.hypot(edge_x, edge_y)
    edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255

    return edge_x.astype(np.uint8), edge_y.astype(np.uint8), edge_magnitude.astype(np.uint8)


def convolve_2d(image, kernel):

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




def compute_harris_response(I_x, I_y, k=0.04):
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


def mark_corners(img, harris_response, threshold_ratio=0.95):
    """Mark detected corners on the image."""
    img_marked = img.copy()
    threshold = threshold_ratio * harris_response.max()
    img_marked[harris_response > threshold] = [0, 0, 255]  # Mark corners in red
    return img_marked


def apply_harris_corner_detection():
    """Full pipeline for Harris Corner Detection."""
    global img, img_gray, img_display

    if img is None:
        return

    I_x, I_y = compute_gradients(img_gray)
    harris_response = compute_harris_response(I_x, I_y)
    img_corners = mark_corners(img, harris_response)

    # Resize result image to 250x250
    img_corners_resized = cv2.resize(img_corners, (250, 250))

    # Convert to display format
    img_corners_display = cv2.cvtColor(img_corners_resized, cv2.COLOR_BGR2RGB)
    img_corners_display = ImageTk.PhotoImage(Image.fromarray(img_corners_display))

    # Show image in right panel
    label_output.config(image=img_corners_display)
    label_output.image = img_corners_display
# GUI setup
root = tk.Tk()
root.title("Harris Corner Detector")
root.geometry("650x450")
root.configure(bg="#f0f0f0")

# Title Label
title_label = tk.Label(root, text="Harris Corner Detector", font=("Arial", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

# Frame for Buttons
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack()

btn_load = tk.Button(button_frame, text="Load Image", command=load_image, font=("Arial", 12), bg="#007bff", fg="white")
btn_load.grid(row=0, column=0, padx=10, pady=5)

btn_harris = tk.Button(button_frame, text="Apply Harris Detector", command=apply_harris_corner_detection, font=("Arial", 12), bg="#28a745", fg="white")
btn_harris.grid(row=0, column=1, padx=10, pady=5)

time_label = tk.Label(root, text="Computation Time: -- sec", font=("Arial", 12), bg="#f0f0f0")
time_label.pack(pady=5)

# Frame for Image Displays
image_frame = tk.Frame(root, bg="#f0f0f0")
image_frame.pack()

# Placeholder Image
placeholder = ImageTk.PhotoImage(Image.new("RGB", (300, 300), (200, 200, 200)))

label_input = tk.Label(image_frame, image=placeholder, bg="white", width=300, height=300)
label_input.grid(row=0, column=0, padx=10, pady=10)
label_input.image = placeholder
label_input_text = tk.Label(image_frame, text="Original Image", font=("Arial", 12), bg="#f0f0f0")
label_input_text.grid(row=1, column=0, pady=5)

label_output = tk.Label(image_frame, image=placeholder, bg="white", width=300, height=300)
label_output.grid(row=0, column=1, padx=10, pady=10)
label_output.image = placeholder
label_output_text = tk.Label(image_frame, text="Processed Image", font=("Arial", 12), bg="#f0f0f0")
label_output_text.grid(row=1, column=1, pady=5)

# Run GUI
root.mainloop()
