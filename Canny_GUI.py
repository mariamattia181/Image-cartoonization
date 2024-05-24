import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Canvas, Checkbutton, BooleanVar
from PIL import Image, ImageTk
from scipy.spatial.distance import cdist

CARTOON_PALETTE_RGB = np.array([
    [255, 255, 255],  # White
    [0, 0, 0],        # Black
    [255, 0, 0],      # Red
    [0, 255, 0],      # Green
    [0, 0, 255],      # Blue
    [255, 255, 0],    # Yellow
    [255, 165, 0],    # Orange
    [128, 0, 128],    # Purple
    [0, 255, 255],    # Cyan
    [255, 20, 147],   # Deep Pink
    [173, 216, 230],  # Light Blue
    [255, 182, 193],  # Light Pink
    [144, 238, 144],  # Light Green
    [255, 228, 196],  # Bisque
    [255, 218, 185],  # Peach Puff
    [135, 206, 250],  # Light Sky Blue
    [240, 128, 128],  # Light Coral
    [245, 222, 179],  # Wheat
    [221, 160, 221],  # Plum
    [255, 105, 180]   # Hot Pink
], dtype=np.uint8)

CARTOON_PALETTE_LAB = cv2.cvtColor(CARTOON_PALETTE_RGB[np.newaxis, :, :], cv2.COLOR_RGB2LAB).squeeze()

def apply_custom_cartoon_palette(image, palette_rgb, palette_lab):
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    data_lab = image_lab.reshape((-1, 3))
    distances = cdist(data_lab, palette_lab, metric='euclidean')
    nearest_palette_indices = np.argmin(distances, axis=1)
    quantized_lab = palette_lab[nearest_palette_indices].reshape(image_lab.shape)
    quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)

    return quantized_bgr

def cartoonize_image(img, k=8, line_thickness=1, saturation=1.0, intensity=1.0, use_cartoon_palette=False):
    
    def quantize(img, k):  # kmeans quantization
        img = cv2.medianBlur(img, 5)  # Noise removal
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)

        return result

    def detect_edge(img, thickness):  # Canny edge detection
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)
        edges = cv2.Canny(img_gray, 100, 200)
        edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8))
        edges = cv2.bitwise_not(edges)

        return edges

    color = quantize(img, k)
    edges = detect_edge(img, line_thickness)
    edges_resized = cv2.resize(edges, (color.shape[1], color.shape[0]))
    edges_resized_colored = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
    combined_image = cv2.bitwise_and(color, edges_resized_colored)
    hsv = cv2.cvtColor(combined_image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * saturation
    combined_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    combined_image = cv2.bilateralFilter(combined_image, 9, 75 * intensity, 75 * intensity)
    if use_cartoon_palette:
        combined_image = apply_custom_cartoon_palette(combined_image, CARTOON_PALETTE_RGB, CARTOON_PALETTE_LAB)

    return combined_image

class CartoonizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Cartoonizer")
        self.image_path = None

        self.canvas_width = 1200
        self.canvas_height = 700

        self.canvas = Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.cartoon_button = tk.Button(root, text="Cartoonize", command=self.cartoonize)
        self.cartoon_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.num_colors = tk.Scale(root, from_=2, to_=20, orient='horizontal', label='Number of Colors')
        self.num_colors.set(8)
        self.num_colors.pack(side=tk.LEFT, padx=10, pady=10)

        self.line_thickness = tk.Scale(root, from_=1, to_=10, orient='horizontal', label='Line Thickness')
        self.line_thickness.set(1)
        self.line_thickness.pack(side=tk.LEFT, padx=10, pady=10)

        self.saturation = tk.Scale(root, from_=0.1, to_=2.0, resolution=0.1, orient='horizontal', label='Color Saturation')
        self.saturation.set(1.0)
        self.saturation.pack(side=tk.LEFT, padx=10, pady=10)

        self.intensity = tk.Scale(root, from_=0.1, to_=2.0, resolution=0.1, orient='horizontal', label='Stylization Intensity')
        self.intensity.set(1.0)
        self.intensity.pack(side=tk.LEFT, padx=10, pady=10)

        self.use_cartoon_palette_var = BooleanVar()
        self.use_cartoon_palette_checkbox = Checkbutton(root, text="Use Cartoon Palette", variable=self.use_cartoon_palette_var)
        self.use_cartoon_palette_checkbox.pack(side=tk.LEFT, padx=10, pady=10)

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.original_image)

    def display_image(self, image):
        image_height, image_width = image.shape[:2]
        aspect_ratio = image_width / image_height

        if image_width > self.canvas_width or image_height > self.canvas_height:
            if aspect_ratio > 1:
                new_width = self.canvas_width
                new_height = int(self.canvas_width / aspect_ratio)
            else:
                new_height = self.canvas_height
                new_width = int(self.canvas_height * aspect_ratio)
        else:
            new_width, new_height = image_width, image_height

        resized_image = cv2.resize(image, (new_width, new_height))

        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.canvas.image = image_tk

    def cartoonize(self):
        if self.image_path:
            num_colors = self.num_colors.get()
            line_thickness = self.line_thickness.get()
            saturation = self.saturation.get()
            intensity = self.intensity.get()
            use_cartoon_palette = self.use_cartoon_palette_var.get()
            cartoon_image = cartoonize_image(self.original_image, k=num_colors, line_thickness=line_thickness, saturation=saturation, intensity=intensity, use_cartoon_palette=use_cartoon_palette)
            self.display_image(cartoon_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = CartoonizerApp(root)
    root.mainloop()
