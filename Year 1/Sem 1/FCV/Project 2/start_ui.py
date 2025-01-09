import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from pathlib import Path
import cv2
import datetime
import os

class YOLODetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Detection Interface")
        
        self.model = YOLO("runs-session-4-scratch-FINAL/runs-session-4-scratch/train/exp_1/weights/best.pt")
        self.conf_threshold = 0.6

        tk.Button(root, text="Select File", command=self.select_file).pack(pady=5)
        tk.Button(root, text="Select Directory", command=self.select_directory).pack(pady=5)
        tk.Button(root, text="Use Webcam", command=self.use_webcam).pack(pady=5)

    def create_output_dir(self, type_dir):
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        output_dir = Path("inference_results") / timestamp / type_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def process_image(self, image_path, output_dir):
        try:
            print(f"Processing {Path(image_path).name}...")
            img = cv2.imread(str(image_path))
            
            results = self.model.predict(
                source=img,
                show=False,
                stream=True,
                conf=self.conf_threshold
            )
            
            for r in results:
                plotted_img = r.plot()
                output_path = str(output_dir / f'detected_{Path(image_path).name}')
                cv2.imwrite(output_path, plotted_img)
                
            print(f"✓ Successfully processed {Path(image_path).name}")
        except Exception as e:
            print(f"❌ Error processing {Path(image_path).name}: {str(e)}")

    def process_video(self, video_path, output_dir):
        try:
            print(f"Processing {Path(video_path).name}...")
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise Exception("Could not open video file")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            max_display_width = 480
            max_display_height = 360
            
            width_scale = max_display_width / width
            height_scale = max_display_height / height
            
            scale = min(width_scale, height_scale)
            
            display_width = int(width * scale)
            display_height = int(height * scale)
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            output_path = str(output_dir / f'detected_{Path(video_path).name}')
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            
            window_name = 'Video Detection (Press Q to quit)'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                results = self.model.predict(
                    source=frame,
                    show=False,
                    stream=True,
                    conf=self.conf_threshold
                )
                
                for r in results:
                    plotted_frame = r.plot()
                    writer.write(plotted_frame)
                    
                    display_frame = cv2.resize(plotted_frame, (display_width, display_height))
                    cv2.imshow(window_name, display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            print(f"✓ Successfully processed {Path(video_path).name}")
            
        except Exception as e:
            print(f"❌ Error processing {Path(video_path).name}: {str(e)}")
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Media files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")]
        )
        if file_path:
            file_type = "videos" if file_path.lower().endswith(('.mp4', '.avi', '.mov')) else "images"
            output_dir = self.create_output_dir(file_type)
            
            if file_type == "images":
                self.process_image(file_path, output_dir)
            else:
                self.process_video(file_path, output_dir)
            
            messagebox.showinfo("Success", "Processing complete!")

    def select_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            images_output = self.create_output_dir("images")
            videos_output = self.create_output_dir("videos")
            
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.mp4', '*.avi', '*.mov']:
                files = list(Path(dir_path).glob(ext))
                for file_path in files:
                    if file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                        self.process_video(file_path, videos_output)
                    else:
                        self.process_image(file_path, images_output)
            
            messagebox.showinfo("Success", "Directory processing complete!")

    def use_webcam(self):
        output_dir = self.create_output_dir("videos")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam!")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = str(output_dir / 'webcam_detection.mp4')
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        window_name = 'Webcam Detection (Press Q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                results = self.model.predict(
                    source=frame,
                    show=False,
                    stream=True,
                    conf=self.conf_threshold
                )
                
                for r in results:
                    plotted_frame = r.plot()
                    writer.write(plotted_frame)
                    cv2.imshow(window_name, plotted_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
        except Exception as e:
            messagebox.showerror("Error", f"Webcam processing error: {str(e)}")
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectionUI(root)
    root.mainloop()
