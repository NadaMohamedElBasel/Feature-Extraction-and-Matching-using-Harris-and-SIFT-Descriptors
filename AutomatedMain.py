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
def apply_harris():
    global img, img_gray, img_result

    if img is None:
        return

    # Start time measurement
    start_time = time.time()

    # Convert to float and apply Harris corner detection
    gray_float = np.float32(img_gray)
    harris_response = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

    # Mark the corners
    threshold = 0.01 * harris_response.max()
    img_result = img.copy()
    img_result[harris_response > threshold] = [0, 0, 255]  # Mark corners in red

    # End time measurement
    execution_time = time.time() - start_time
    time_label.config(text=f"Computation Time: {execution_time:.6f} sec")

    # Convert to display format
    img_result_display = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    img_result_display = ImageTk.PhotoImage(Image.fromarray(cv2.resize(img_result_display, (300, 300))))

    # Show image in right panel
    label_output.config(image=img_result_display)
    label_output.image = img_result_display

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

btn_harris = tk.Button(button_frame, text="Apply Harris Detector", command=apply_harris, font=("Arial", 12), bg="#28a745", fg="white")
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