import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.applications import MobileNetV2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow.keras.backend as K
import cv2
import threading
import time
from datetime import datetime

def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss function for Siamese networks
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    margin = 0.2
    y_pred = tf.maximum(y_pred, 1e-7)
    y_pred_norm = y_pred / tf.reduce_mean(y_pred)
    similar_loss = y_true * tf.square(y_pred_norm)
    dissimilar_loss = (1.0 - y_true) * tf.square(tf.maximum(margin - y_pred_norm, 0.0))
    loss = tf.reduce_mean(similar_loss + dissimilar_loss)
    return 0.1 * loss

def create_base_network(input_shape):
    """Create the base network for feature extraction"""
    base_network = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze almost all layers (leave only top 10 trainable)
    for layer in base_network.layers[:-10]:
        layer.trainable = False
    
    # Very simple feature extraction
    x = base_network.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Strong regularization
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Single dense layer with L2 regularization
    x = tf.keras.layers.Dense(
        128, 
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Final embedding with L2 regularization
    embedding_layer = tf.keras.layers.Dense(
        64,
        activation=None,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    embedding_layer = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1)
    )(embedding_layer)
    
    return tf.keras.models.Model(inputs=base_network.input, outputs=embedding_layer, name='base_network')

def euclidean_distance(vectors):
    """Calculate the Euclidean distance between two vectors."""
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, 1e-7))

class EnhancedSiameseTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Siamese Study")
        self.root.geometry("1200x700")
        
        # Model variables
        self.model = None
        self.model_path = None
        self.threshold = 0.75  # Default threshold
        self.img1_path = None
        self.img2_path = None
        
        # Camera variables
        self.camera_active = False
        self.camera_thread = None
        self.cap = None
        self.current_camera_frame = None
        self.camera_target = None
        
        # Create UI
        self.create_ui()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_ui(self):
        # Set style
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10))
        style.configure("Model.TButton", background="#2196F3", foreground="white")
        style.configure("Camera.TButton", background="#4CAF50")
        style.configure("Compare.TButton", background="#FF9800", foreground="white")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Siamese Study", 
            font=("Arial", 24, "bold")
        )
        title_label.pack(pady=(5, 15))
        
        # Model selection section
        model_frame = ttk.LabelFrame(main_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill="x", pady=10)
        
        model_info_frame = ttk.Frame(model_frame)
        model_info_frame.pack(fill="x")
        
        self.model_btn = ttk.Button(
            model_info_frame, 
            text="Load Model", 
            command=self.load_model_file,
            style="Model.TButton"
        )
        self.model_btn.pack(side="left", padx=5)
        
        self.model_status = ttk.Label(
            model_info_frame, 
            text="No model loaded",
            font=("Arial", 10, "italic"),
            foreground="black"
        )
        self.model_status.pack(side="left", padx=10)
        
        # Auto-load button for existing models
        self.auto_load_btn = ttk.Button(
            model_info_frame,
            text="Auto-Load Latest",
            command=self.auto_load_model
        )
        self.auto_load_btn.pack(side="right", padx=5)
        
        # Main content area - Split into top and bottom sections
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill="both", expand=True, pady=5)
        
        # Top section - Images and controls in three columns
        # Left column - Image 1
        left_column = ttk.LabelFrame(top_frame, text="Image 1 (Reference)", padding=10)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.img1_canvas = tk.Canvas(left_column, width=280, height=280, bg="lightgray")
        self.img1_canvas.pack(pady=(0, 10))
        
        img1_btn_frame = ttk.Frame(left_column)
        img1_btn_frame.pack()
        
        self.img1_btn = ttk.Button(img1_btn_frame, text="Browse Image", command=self.load_image1)
        self.img1_btn.pack(pady=2)
        
        img1_camera_frame = ttk.Frame(left_column)
        img1_camera_frame.pack(pady=(5, 0))
        
        self.img1_camera_btn = ttk.Button(img1_camera_frame, text="Camera", command=lambda: self.toggle_camera(1))
        self.img1_camera_btn.pack(side="left", padx=2)
        
        self.img1_capture_btn = ttk.Button(img1_camera_frame, text="Capture", command=lambda: self.capture_image(1), state="disabled")
        self.img1_capture_btn.pack(side="left", padx=2)
        
        # Middle column - Controls and Analysis
        middle_column = ttk.Frame(top_frame)
        middle_column.pack(side="left", fill="y", padx=10)
        
        # Compare button section
        control_frame = ttk.LabelFrame(middle_column, text="Analysis Control", padding=15)
        control_frame.pack(fill="x", pady=(0, 10))
        
        self.compare_btn = ttk.Button(control_frame, text="Compare Images", command=self.compare_images, state="disabled")
        self.compare_btn.pack()
        
        # Threshold control - moved up to be more visible
        threshold_frame = ttk.LabelFrame(middle_column, text="Decision Threshold", padding=15)
        threshold_frame.pack(fill="x", pady=(0, 10))
        
        threshold_info = ttk.Label(threshold_frame, text="Minimum similarity required (50%-95%):", font=("Arial", 9))
        threshold_info.pack(pady=(0, 5))
        
        threshold_control_frame = ttk.Frame(threshold_frame)
        threshold_control_frame.pack()
        
        self.threshold_var = tk.DoubleVar(value=self.threshold)
        self.threshold_scale = ttk.Scale(
            threshold_control_frame,
            from_=0.5,
            to=0.95,
            orient="horizontal",
            variable=self.threshold_var,
            length=200,
            command=self.update_threshold
        )
        self.threshold_scale.pack(side="left", padx=(0, 10))
        
        self.threshold_label = ttk.Label(threshold_control_frame, text=f"{self.threshold:.2f}", font=("Arial", 12, "bold"))
        self.threshold_label.pack(side="left")
        
        # Analysis Results - prominently placed
        result_frame = ttk.LabelFrame(middle_column, text="Analysis Results", padding=15)
        result_frame.pack(fill="both", expand=True)
        
        self.result_label = ttk.Label(
            result_frame, 
            text="Load a model and two images,\nthen click 'Compare Images'",
            font=("Arial", 11),
            justify="center"
        )
        self.result_label.pack(pady=(0, 15))
        
        # Similarity score display
        similarity_info = ttk.Label(result_frame, text="Similarity Score:", font=("Arial", 9, "bold"))
        similarity_info.pack()
        
        score_frame = ttk.Frame(result_frame)
        score_frame.pack(fill="x", pady=(5, 0))
        
        self.distance_var = tk.DoubleVar(value=0)
        self.distance_bar = ttk.Progressbar(
            score_frame, 
            orient="horizontal", 
            length=200, 
            mode="determinate",
            variable=self.distance_var
        )
        self.distance_bar.pack(side="left", padx=(0, 10))
        
        self.distance_label = ttk.Label(score_frame, text="0.00%", font=("Arial", 12, "bold"))
        self.distance_label.pack(side="left")
        
        # Right column - Image 2
        right_column = ttk.LabelFrame(top_frame, text="Image 2 (Comparison)", padding=10)
        right_column.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        self.img2_canvas = tk.Canvas(right_column, width=280, height=280, bg="lightgray")
        self.img2_canvas.pack(pady=(0, 10))
        
        img2_btn_frame = ttk.Frame(right_column)
        img2_btn_frame.pack()
        
        self.img2_btn = ttk.Button(img2_btn_frame, text="Browse Image", command=self.load_image2)
        self.img2_btn.pack(pady=2)
        
        img2_camera_frame = ttk.Frame(right_column)
        img2_camera_frame.pack(pady=(5, 0))
        
        self.img2_camera_btn = ttk.Button(img2_camera_frame, text="Camera", command=lambda: self.toggle_camera(2))
        self.img2_camera_btn.pack(side="left", padx=2)
        
        self.img2_capture_btn = ttk.Button(img2_camera_frame, text="Capture", command=lambda: self.capture_image(2), state="disabled")
        self.img2_capture_btn.pack(side="left", padx=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Please load a model and images.")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 9)
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model_file(self):
        """Load a Siamese model from file"""
        file_path = filedialog.askopenfilename(
            title="Select Siamese Model File",
            filetypes=[("HDF5 files", "*.h5"), ("SavedModel", "*.pb"), ("All files", "*.*")],
            initialdir="./models"
        )
        
        if file_path:
            self.load_model_from_path(file_path)
    
    def auto_load_model(self):
        """Auto-load the latest model from the models directory"""
        try:
            model_dir = "./models"
            if not os.path.exists(model_dir):
                messagebox.showerror("Error", "Models directory not found!")
                return
            
            # Look for Siamese models
            siamese_models = [f for f in os.listdir(model_dir) if f.startswith('siamese_') and f.endswith('.h5')]
            if not siamese_models:
                siamese_models = [f for f in os.listdir(model_dir) if 'siamese' in f.lower() and f.endswith('.h5')]
            
            if not siamese_models:
                messagebox.showwarning("No Models", "No Siamese models found in the models directory!")
                return
            
            # Get the latest model
            latest_model = max(siamese_models, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
            model_path = os.path.join(model_dir, latest_model)
            
            self.load_model_from_path(model_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to auto-load model: {str(e)}")
    
    def load_model_from_path(self, model_path):
        """Load a model from the specified path"""
        try:
            self.status_var.set("Loading model...")
            print(f"Loading model from: {model_path}")
            
            # Check file size
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"Model size: {size_mb:.2f} MB")
            
            # Create the Siamese network architecture
            input_shape = (224, 224, 3)
            
            # Create the base network
            embedding_model = create_base_network(input_shape)
            print("Base network created")
            
            # Create Siamese network inputs
            input_a = tf.keras.layers.Input(shape=input_shape)
            input_b = tf.keras.layers.Input(shape=input_shape)
            
            # Process inputs through the embedding model
            processed_a = embedding_model(input_a)
            processed_b = embedding_model(input_b)
            print("Created embedding layers")
            
            # Calculate distance
            distance = tf.keras.layers.Lambda(
                euclidean_distance,
                name='distance'
            )([processed_a, processed_b])
            print("Added distance layer")
            
            # Create the model
            self.model = tf.keras.models.Model(inputs=[input_a, input_b], outputs=distance)
            
            # Load weights
            print("Loading weights...")
            self.model.load_weights(model_path)
            print("Weights loaded successfully!")
            
            self.model_path = model_path
            
            # Try to load threshold from same directory
            model_dir = os.path.dirname(model_path)
            threshold_path = os.path.join(model_dir, "best_threshold.txt")
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    self.threshold = float(f.read().strip())
                    self.threshold_var.set(self.threshold)
                    self.threshold_label.config(text=f"{self.threshold:.2f}")
                print(f"Loaded threshold value: {self.threshold}")
            
            # Update UI
            model_name = os.path.basename(model_path)
            self.model_status.config(text=f"Loaded: {model_name}")
            self.status_var.set(f"Model loaded successfully: {model_name}")
            self.update_compare_button()
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Model Load Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model. Check console for details.")
    
    def load_image1(self):
        """Load the first image from file"""
        path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if path:
            self.img1_path = path
            self.display_image(path, self.img1_canvas)
            self.update_compare_button()
            self.status_var.set(f"Reference image loaded: {os.path.basename(path)}")
    
    def load_image2(self):
        """Load the second image from file"""
        path = filedialog.askopenfilename(
            title="Select Comparison Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if path:
            self.img2_path = path
            self.display_image(path, self.img2_canvas)
            self.update_compare_button()
            self.status_var.set(f"Comparison image loaded: {os.path.basename(path)}")
    
    def toggle_camera(self, target):
        """Toggle camera on/off for the specified target (1 or 2)"""
        if self.camera_active:
            self.stop_camera()
            return
        
        self.camera_target = target
        self.start_camera()
        
        # Update UI
        if target == 1:
            self.img1_camera_btn.config(text="Stop Camera")
            self.img1_capture_btn.config(state="normal")
            self.img2_camera_btn.config(state="disabled")
        else:
            self.img2_camera_btn.config(text="Stop Camera")
            self.img2_capture_btn.config(state="normal")
            self.img1_camera_btn.config(state="disabled")
    
    def start_camera(self):
        """Start the camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.camera_active = True
            self.camera_thread = threading.Thread(target=self.update_camera_feed)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            self.status_var.set("Camera active. Press 'Capture' to take a photo.")
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera capture"""
        self.camera_active = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        # Reset UI
        if self.camera_target == 1:
            self.img1_camera_btn.config(text="Camera")
            self.img1_capture_btn.config(state="disabled")
            self.img2_camera_btn.config(state="normal")
        else:
            self.img2_camera_btn.config(text="Camera")
            self.img2_capture_btn.config(state="disabled")
            self.img1_camera_btn.config(state="normal")
        
        self.camera_target = None
        self.status_var.set("Camera stopped.")
    
    def update_camera_feed(self):
        """Update the camera feed in a separate thread"""
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_camera_frame = frame_rgb
            
            # Resize for display
            height, width = frame_rgb.shape[:2]
            max_size = 280
            scale = min(max_size / width, max_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            img = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            canvas = self.img1_canvas if self.camera_target == 1 else self.img2_canvas
            
            # We need to use the main thread to update the UI
            self.root.after(1, lambda: self.update_canvas(canvas, photo, new_width, new_height))
            
            time.sleep(0.03)  # ~30 FPS
    
    def update_canvas(self, canvas, photo, width, height):
        """Update the canvas with the new frame (called from main thread)"""
        if not self.camera_active:
            return
            
        canvas.delete("all")
        canvas.create_image(width//2, height//2, image=photo)
        canvas.image = photo  # Keep a reference
    
    def capture_image(self, target):
        """Capture an image from the camera"""
        if not self.camera_active or self.current_camera_frame is None:
            messagebox.showerror("Error", "Camera is not active")
            return
        
        # Create a temporary file to save the captured image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = os.path.join(temp_dir, f"capture_{timestamp}.jpg")
        
        # Save the current frame
        img = Image.fromarray(self.current_camera_frame)
        img.save(temp_file)
        
        # Update the appropriate image path
        if target == 1:
            self.img1_path = temp_file
        else:
            self.img2_path = temp_file
        
        # Stop the camera
        self.stop_camera()
        
        # Display the captured image
        self.display_image(temp_file, self.img1_canvas if target == 1 else self.img2_canvas)
        self.update_compare_button()
        
        self.status_var.set(f"Image {target} captured and saved.")
    
    def display_image(self, path, canvas):
        """Display an image on the specified canvas"""
        try:
            # Load and resize image
            img = Image.open(path)
            img = img.resize((280, 280), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Display on canvas
            canvas.delete("all")
            canvas.create_image(140, 140, image=photo)
            canvas.image = photo  # Keep a reference
        except Exception as e:
            print(f"Error displaying image: {e}")
            messagebox.showerror("Image Error", f"Failed to display image: {str(e)}")
    
    def update_compare_button(self):
        """Enable the compare button if model and both images are loaded"""
        if self.model and self.img1_path and self.img2_path:
            self.compare_btn.config(state="normal")
        else:
            self.compare_btn.config(state="disabled")
    
    def update_threshold(self, value):
        """Update threshold value when slider changes"""
        self.threshold = float(value)
        self.threshold_label.config(text=f"{self.threshold:.2f}")
        
        # If we have already compared images, update the result
        if hasattr(self, 'current_distance') and self.current_distance is not None:
            self.update_result_display(self.current_distance)
    
    def preprocess_image(self, img_path):
        """Load and preprocess an image for the model"""
        try:
            # Get image dimensions from model input
            input_shape = self.model.input_shape[0][1:3]
            
            # Load and resize image
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=input_shape)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Additional normalization to match training
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            print(f"Error preprocessing image {img_path}: {e}")
            raise
    
    def compare_images(self):
        """Compare the two loaded images using the Siamese network"""
        if not self.model:
            messagebox.showerror("Error", "No model loaded!")
            return
        
        if not self.img1_path or not self.img2_path:
            messagebox.showerror("Error", "Please load both images first!")
            return
        
        try:
            self.status_var.set("Processing images...")
            self.root.update()
            
            # Preprocess images
            img1 = self.preprocess_image(self.img1_path)
            img2 = self.preprocess_image(self.img2_path)
            
            # Predict using the model
            distance = self.model.predict([img1, img2], verbose=0)[0][0]
            self.current_distance = distance
            
            # Update the interface with the result
            self.update_result_display(distance)
            
        except Exception as e:
            print(f"Error comparing images: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Comparison Error", f"Error comparing images: {str(e)}")
            self.status_var.set("Error during comparison. See error message.")
    
    def update_result_display(self, distance):
        """Update the display with the comparison result"""
        # Calculate similarity score (inverse of distance, scaled to 0-100)
        similarity_score = max(0, min(100, (1 - distance) * 100))
        
        # Update progress bar
        self.distance_var.set(similarity_score)
        self.distance_label.config(text=f"{similarity_score:.1f}%")
        
        # Convert threshold (0.1-1.0) to similarity percentage (10%-100%)
        threshold_percentage = self.threshold * 100
        
        # Compare similarity score with threshold percentage
        is_similar = similarity_score >= threshold_percentage
        
        if is_similar:
            result_text = "HIGH SIMILARITY DETECTED"
            result_detail = f"The images appear to show the SAME animal"
            color = "#4CAF50"  # Green
            emoji = "✓"
        else:
            result_text = "HIGH DIFFERENCE DETECTED"
            result_detail = f"The images appear to show DIFFERENT animals"
            color = "#F44336"  # Red
            emoji = "✗"
        
        detailed_info = f"Distance: {distance:.4f} | Threshold: {threshold_percentage:.1f}%\nSimilarity Score: {similarity_score:.1f}%"
        
        self.result_label.config(
            text=f"{emoji} {result_text}\n{result_detail}\n{detailed_info}",
            foreground=color,
            font=("Arial", 12, "bold")
        )
        
        self.status_var.set(f"Analysis complete - {result_text.lower()}")
    
    def on_closing(self):
        """Handle window closing event"""
        if self.camera_active:
            self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    try:
        print("Starting Enhanced Siamese Tester...")
        root = tk.Tk()
        app = EnhancedSiameseTester(root)
        root.mainloop()
        print("Application closed normally")
    except Exception as e:
        print(f"Application crashed with error: {str(e)}")
        import traceback
        traceback.print_exc() 