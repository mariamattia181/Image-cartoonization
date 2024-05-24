import tkinter as tk
from tkinter import Toplevel

class MainPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Page")

        self.welcome_label = tk.Label(root, text="Welcome to the Image Cartoonization Application!", font=("Arial", 18))
        self.welcome_label.pack(pady=20)

        self.cartoonizer_button = tk.Button(root, text="Canny Edge Detection", command=self.open_canny)
        self.cartoonizer_button.pack(pady=10)

        self.other_button1 = tk.Button(root, text="Adaptive Threshold Edge Detection", command=self.open_adaptive)
        self.other_button1.pack(pady=10)

        self.other_button2 = tk.Button(root, text="Morphological Edge Detection", command=self.open_morphological)
        self.other_button2.pack(pady=10)

        self.other_button3 = tk.Button(root, text="Sobel Edge Detection", command=self.open_sobel)
        self.other_button3.pack(pady=10)

    def open_canny(self):
        new_window = Toplevel(self.root)
        import Canny_GUI 
        app = Canny_GUI.CartoonizerApp(new_window)

    def open_adaptive(self):
        new_window = Toplevel(self.root)
        import Adaptive_GUI
        app = Adaptive_GUI.CartoonizerApp(new_window)

    def open_morphological(self):
        new_window = Toplevel(self.root)
        import Morphological_GUI
        app = Morphological_GUI.CartoonizerApp(new_window)

    def open_sobel(self):
        new_window = Toplevel(self.root)
        import sobel_GUI
        app = sobel_GUI.CartoonizerApp(new_window)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainPage(root)
    root.mainloop()
