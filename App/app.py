import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import math

class HandGestureAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Analyzer")
        self.root.geometry("800x600")
        
        # Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame untuk kontrol
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        
        # Tombol untuk memilih gambar
        self.select_btn = tk.Button(
            control_frame, 
            text="Pilih Gambar", 
            command=self.load_image,
            bg="#4CAF50",
            fg="white",
            font=('Arial', 12))
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        # Tombol untuk analisis
        self.analyze_btn = tk.Button(
            control_frame,
            text="Analisis Gesture",
            command=self.analyze_gesture,
            bg="#2196F3",
            fg="white",
            font=('Arial', 12),
            state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Label untuk hasil
        self.result_label = tk.Label(
            control_frame,
            text="Gesture: -",
            font=('Arial', 14, 'bold'),
            fg="#333")
        self.result_label.pack(side=tk.LEFT, padx=20)
        
        # Frame untuk menampilkan gambar
        self.image_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label untuk gambar asli
        self.original_label = tk.Label(
            self.image_frame, 
            text="Gambar Asli", 
            bg="#f0f0f0",
            font=('Arial', 10))
        self.original_label.pack()
        
        self.original_img_label = tk.Label(self.image_frame)
        self.original_img_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Label untuk gambar dengan deteksi
        self.detected_label = tk.Label(
            self.image_frame, 
            text="Deteksi Tangan", 
            bg="#f0f0f0",
            font=('Arial', 10))
        self.detected_label.pack()
        
        self.detected_img_label = tk.Label(self.image_frame)
        self.detected_img_label.pack(side=tk.LEFT, padx=10, pady=10)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        
        if file_path:
            try:
                self.image_path = file_path
                
                # Buka gambar dengan PIL
                self.pil_image = Image.open(file_path)
                
                # Tampilkan gambar asli
                self.display_original_image()
                
                # Aktifkan tombol analisis
                self.analyze_btn.config(state=tk.NORMAL)
                self.result_label.config(text="Gesture: -")
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat gambar: {str(e)}")
    
    def display_original_image(self):
        # Resize gambar untuk ditampilkan di GUI
        max_size = (400, 400)
        pil_image_resized = self.pil_image.copy()
        pil_image_resized.thumbnail(max_size, Image.LANCZOS)
        
        # Konversi ke format Tkinter
        self.tk_original_image = ImageTk.PhotoImage(pil_image_resized)
        self.original_img_label.config(image=self.tk_original_image)
        
        # Kosongkan gambar deteksi
        self.detected_img_label.config(image='')
    
    def analyze_gesture(self):
        if hasattr(self, 'image_path'):
            try:
                # Baca gambar dengan OpenCV
                image = cv2.imread(self.image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Proses deteksi tangan
                results = self.hands.process(image_rgb)
                
                # Salin gambar untuk ditandai
                annotated_image = image.copy()
                
                gesture = "Tidak terdeteksi"
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Gambar landmark tangan
                        mp.solutions.drawing_utils.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS)
                        
                        # Deteksi gesture
                        gesture = self.detect_gesture(hand_landmarks.landmark)
                
                # Update hasil
                self.result_label.config(text=f"Gesture: {gesture}")
                
                # Tampilkan gambar dengan deteksi
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                pil_annotated = Image.fromarray(annotated_image)
                pil_annotated.thumbnail((400, 400), Image.LANCZOS)
                self.tk_annotated_image = ImageTk.PhotoImage(pil_annotated)
                self.detected_img_label.config(image=self.tk_annotated_image)
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menganalisis gambar: {str(e)}")
    
    def detect_gesture(self, landmarks):
        # Finger tip & pip points
        tip_ids = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky

        # Finger states (1: up, 0: down)
        fingers = []

        # Ibu jari (x axis karena mengarah ke samping)
        if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # Jari lainnya (y axis karena tegak lurus ke atas)
        for i in range(1, 5):
            if landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        # Buat gesture dari pola jari
        if fingers == [0, 0, 0, 0, 0]:
            return "FIST"
        elif fingers == [1, 1, 0, 0, 0]:
            return "POINTING"
        elif fingers == [1, 1, 0, 0, 1]:
            return "ROCK"
        elif fingers == [1, 0, 0, 0, 0]:
            return "THUMB UP"
        elif fingers == [0, 1, 0, 0, 0]:
            return "ONE"
        elif fingers == [0, 1, 1, 0, 0]:
            return "TWO"
        elif fingers == [0, 1, 1, 1, 0]:
            return "THREE"
        elif fingers == [0, 1, 1, 1, 1]:
            return "FOUR"
        elif fingers == [1, 1, 1, 1, 1]:
            return "FIVE"
        else:
            return "UNKNOWN"

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureAnalyzer(root)
    root.mainloop()