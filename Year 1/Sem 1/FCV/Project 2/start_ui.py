import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import cv2
import os
from pathlib import Path
import numpy as np
import torch

# Initialize YOLO model
model = YOLO("runs-session-1-100-epochs/runs-session-1/train/exp_1/weights/best.pt")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def process_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    
    # Run detection
    results = model(img)
    annotated_frame = results[0].plot()
    
    # Save the annotated image
    output_path = os.path.join(RESULTS_DIR, f"{Path(image_path).stem}_detection{Path(image_path).suffix}")
    cv2.imwrite(output_path, annotated_frame)
    
    # Calculate window size while maintaining aspect ratio
    max_width = 1200
    max_height = 800
    scale = min(max_width / img.shape[1], max_height / img.shape[0])
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    
    # Create window with adjusted size
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", window_width, window_height)
    
    # Display the result
    cv2.imshow("Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create output video file in results directory
    output_path = os.path.join(RESULTS_DIR, f"{Path(video_path).stem}_detection.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        cv2.imshow("Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def start_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        tk.messagebox.showerror("Error", "Could not open webcam!")
        return

    # Set resolution to 640x640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    window_name = "Webcam Detection (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 640)
    
    running = True
    def handle_window_close(event=None):
        nonlocal running
        running = False
    
    # Initialize previous detections for smoothing
    prev_boxes = []
    smooth_factor = 0.7  # Adjust this value between 0 and 1 (higher = more smoothing)
    
    try:
        while running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (640, 640))
            
            # Run detection
            results = model(frame, conf=0.5)  # Add confidence threshold
            
            # Get current boxes
            current_boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # Apply smoothing if we have previous boxes
            if len(prev_boxes) > 0 and len(current_boxes) == len(prev_boxes):
                smoothed_boxes = current_boxes * (1 - smooth_factor) + prev_boxes * smooth_factor
                # Draw smoothed boxes manually
                annotated_frame = frame.copy()
                for box in smoothed_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # Use original annotations if smoothing isn't possible
                annotated_frame = results[0].plot()
            
            # Store current boxes for next frame
            prev_boxes = current_boxes.copy()
            
            cv2.imshow(window_name, annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

def browse_file():
    path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[
            ("All supported", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"),
            ("Image files", "*.jpg *.jpeg *.png"),
            ("Video files", "*.mp4 *.avi *.mov"),
        ]
    )
    
    if not path:
        return
        
    if path.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image(path)
    elif path.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(path)

def browse_directory():
    path = filedialog.askdirectory(title="Select Directory")
    
    if not path:
        return
        
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(file_path)
        elif file.lower().endswith(('.mp4', '.avi', '.mov')):
            process_video(file_path)

# Update GUI creation
root = tk.Tk()
root.title("YOLO Detection")
root.geometry("300x200")

file_button = tk.Button(root, text="Select File", command=browse_file)
file_button.pack(pady=10)

directory_button = tk.Button(root, text="Select Directory", command=browse_directory)
directory_button.pack(pady=10)

webcam_button = tk.Button(root, text="Webcam", command=start_webcam)
webcam_button.pack(pady=10)

root.mainloop()