import cv2
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import random
import time
import mediapipe as mp

class GestureWars:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Wars - Reign of RPS")
        self.root.geometry("1000x600")
        
        # Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        
        # Inisialisasi kamera
        self.cap = cv2.VideoCapture(0)
        
        # Variabel game
        self.game_active = False
        self.player_choice = None
        self.computer_choice = None
        self.result = None
        self.countdown = 0
        self.score = {"player": 0, "computer": 0, "tie": 0}
        self.last_update_time = 0
        
        # Setup UI
        self.setup_ui()
        
        # Mulai update frame
        self.update_frame()
    
    def setup_ui(self):
        # Frame utama
        self.main_frame = tk.Frame(self.root, bg="#2c3e50")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header (Score)
        self.header_frame = tk.Frame(self.main_frame, bg="#34495e", height=80)
        self.header_frame.pack(fill=tk.X)
        
        self.score_font = font.Font(family="Helvetica", size=16, weight="bold")
        
        self.player_score_label = tk.Label(
            self.header_frame, 
            text="Player: 0", 
            font=self.score_font, 
            bg="#34495e", 
            fg="#ecf0f1")
        self.player_score_label.pack(side=tk.LEFT, padx=20)
        
        self.computer_score_label = tk.Label(
            self.header_frame, 
            text="Computer: 0", 
            font=self.score_font, 
            bg="#34495e", 
            fg="#ecf0f1")
        self.computer_score_label.pack(side=tk.LEFT, padx=20)
        
        self.tie_score_label = tk.Label(
            self.header_frame, 
            text="Tie: 0", 
            font=self.score_font, 
            bg="#34495e", 
            fg="#ecf0f1")
        self.tie_score_label.pack(side=tk.LEFT, padx=20)
        
        # Game area
        self.game_frame = tk.Frame(self.main_frame, bg="#2c3e50")
        self.game_frame.pack(fill=tk.BOTH, expand=True)
        
        # Webcam view
        self.webcam_frame = tk.Frame(self.game_frame, bg="#34495e", width=480, height=480)
        self.webcam_frame.pack_propagate(False)
        self.webcam_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.webcam_label = tk.Label(self.webcam_frame, bg="#34495e")
        self.webcam_label.pack(fill=tk.BOTH, expand=True)
        
        # Computer choice view
        self.computer_frame = tk.Frame(self.game_frame, bg="#34495e", width=480, height=480)
        self.computer_frame.pack_propagate(False)
        self.computer_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.computer_label = tk.Label(
            self.computer_frame, 
            text="Computer\nWaiting...", 
            font=("Helvetica", 24), 
            bg="#34495e", 
            fg="#ecf0f1")
        self.computer_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        self.control_frame = tk.Frame(self.main_frame, bg="#2c3e50", height=80)
        self.control_frame.pack(fill=tk.X)
        
        self.start_button = tk.Button(
            self.control_frame, 
            text="Start Round", 
            command=self.start_round,
            font=("Helvetica", 14),
            bg="#27ae60",
            fg="white",
            activebackground="#2ecc71",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=10)
        self.start_button.pack(pady=10)
        
        self.instruction_label = tk.Label(
            self.control_frame, 
            text="Press 'Start Round' to begin", 
            font=("Helvetica", 12), 
            bg="#2c3e50", 
            fg="#bdc3c7")
        self.instruction_label.pack()
    
    def update_frame_V1(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip dan konversi ke RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Reset player choice setiap frame
            current_gesture = "UNKNOWN"
            
            # Proses deteksi tangan
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Deteksi gesture
                current_gesture = self.detect_gesture(results.multi_hand_landmarks[0].landmark)
                
                # Gambar landmark tangan
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, 
                    results.multi_hand_landmarks[0], 
                    self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
                
                # Jika game aktif, simpan pilihan player
                if self.game_active:
                    current_time = time.time()
                    if current_time - self.last_update_time >= 1:  # Update setiap 1 detik
                        self.last_update_time = current_time
                        self.player_choice = current_gesture
            
            # Tambahkan teks gesture yang terdeteksi
            cv2.putText(
                frame, 
                f"{current_gesture.upper()}", 
                (50, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA)
            
            # Jika game aktif, tambahkan countdown
            if self.game_active:
                remaining_time = max(0, int(self.countdown - time.time()))
                cv2.putText(
                    frame, 
                    f"Time: {remaining_time}", 
                    (frame.shape[1] - 200, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2, 
                    cv2.LINE_AA)
            
            # Convert ke format ImageTk
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update label webcam
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
            
            # Update game state
            self.update_game_state()
        
        # Panggil fungsi ini lagi setelah 10ms
        self.webcam_label.after(10, self.update_frame)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip dan konversi ke RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Dapatkan ukuran frame asli
            original_height, original_width = frame.shape[:2]
            
            # Reset player choice setiap frame
            current_gesture = "UNKNOWN"
            
            # Proses deteksi tangan
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Deteksi gesture
                current_gesture = self.detect_gesture(results.multi_hand_landmarks[0].landmark)
                
                # Gambar landmark tangan
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, 
                    results.multi_hand_landmarks[0], 
                    self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
                
                # Jika game aktif, simpan pilihan player
                if self.game_active:
                    current_time = time.time()
                    if current_time - self.last_update_time >= 1:
                        self.last_update_time = current_time
                        self.player_choice = current_gesture
            
            # Hitung rasio resize berdasarkan ukuran label di Tkinter
            label_width = self.webcam_label.winfo_width()
            label_height = self.webcam_label.winfo_height()
            
            # Jika label belum di-render, gunakan ukuran default
            if label_width <= 1 or label_height <= 1:
                label_width = 480
                label_height = 480
            
            # Hitung faktor skala
            width_scale = label_width / original_width
            height_scale = label_height / original_height
            
            # Tentukan ukuran font dan posisi yang sesuai dengan skala
            font_scale = 1 * min(width_scale, height_scale)
            thickness = max(1, int(2 * min(width_scale, height_scale)))
            
            # Posisi teks yang di-scaled
            text_x = int(10 * width_scale)
            text_y = int(40 * height_scale)
            time_text_x = original_width - int(150 * width_scale)
            
            # Tambahkan teks gesture yang terdeteksi
            cv2.putText(
                frame, 
                f"Current: {current_gesture.upper()}", 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 255, 0), 
                thickness, 
                cv2.LINE_AA)
            
            # Jika game aktif, tambahkan countdown
            if self.game_active:
                remaining_time = max(0, int(self.countdown - time.time()))
                cv2.putText(
                    frame, 
                    f"Time: {remaining_time}", 
                    (time_text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (0, 0, 255), 
                    thickness, 
                    cv2.LINE_AA)
            
            # Resize frame ke ukuran label sebelum konversi ke ImageTk
            resized_frame = cv2.resize(frame, (label_width, label_height))
            
            # Convert ke format ImageTk
            img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update label webcam
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
            
            # Update game state
            self.update_game_state()
        
        # Panggil fungsi ini lagi setelah 10ms
        self.webcam_label.after(10, self.update_frame)

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
            return "rock"
        elif fingers == [0, 1, 1, 0, 0]:
            return "scissors"
        elif fingers == [1, 1, 1, 1, 1]:
            return "paper"
        else:
            return "UNKNOWN"
    
    def start_round(self):
        if not self.game_active:
            self.game_active = True
            self.player_choice = None
            self.computer_choice = random.choice(["rock", "paper", "scissors"])
            self.result = None
            self.countdown = time.time() + 3  # 3 detik countdown
            self.last_update_time = 0
            
            self.start_button.config(state=tk.DISABLED)
            self.instruction_label.config(text="Show your hand now!")
    
    def update_game_state(self):
        if self.game_active:
            current_time = time.time()
            remaining_time = max(0, self.countdown - current_time)
            
            if remaining_time <= 0:
                self.game_active = False
                
                # Jika player tidak memilih, anggap sebagai UNKNOWN
                if self.player_choice is None:
                    self.player_choice = "UNKNOWN"
                
                # Tentukan pemenang
                if self.player_choice == "UNKNOWN":
                    self.result = "invalid"
                else:
                    self.result = self.determine_winner(self.player_choice, self.computer_choice)
                    self.score[self.result] += 1
                
                # Update tampilan
                self.update_score()
                self.show_result()
                
                self.start_button.config(state=tk.NORMAL)
                self.instruction_label.config(text="Press 'Start Round' to play again")
    
    def determine_winner(self, player, computer):
        if player == computer:
            return "tie"
        elif (player == "rock" and computer == "scissors") or \
             (player == "scissors" and computer == "paper") or \
             (player == "paper" and computer == "rock"):
            return "player"
        else:
            return "computer"
    
    def update_score(self):
        self.player_score_label.config(text=f"Player: {self.score['player']}")
        self.computer_score_label.config(text=f"Computer: {self.score['computer']}")
        self.tie_score_label.config(text=f"Tie: {self.score['tie']}")
    
    def show_result(self):
        # Tampilkan pilihan komputer
        emoji_map = {
            "rock": "✊",
            "paper": "✋",
            "scissors": "✌️",
            "UNKNOWN": "❓"
        }
        
        computer_emoji = emoji_map.get(self.computer_choice, "❓")
        player_emoji = emoji_map.get(self.player_choice, "❓")
        
        # Update tampilan komputer
        self.computer_label.config(
            text=f"Computer chose:\n{computer_emoji}\n{self.computer_choice.upper()}",
            font=("Helvetica", 24))
        
        # Tampilkan hasil
        if self.result == "invalid":
            self.instruction_label.config(text="Invalid gesture! Try again")
        else:
            result_text = {
                "player": "You Win!",
                "computer": "Computer Wins!",
                "tie": "It's a Tie!"
            }.get(self.result, "")
            
            self.instruction_label.config(text=f"You chose: {player_emoji}  |  {result_text}")
    
    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureWars(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()