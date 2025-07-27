import cv2
import mediapipe as mp
import numpy as np
import math

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Fungsi untuk menghitung jarak antara dua titik
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Fungsi untuk mendeteksi gesture
def detect_gesture_V1(landmarks):
    # Titik-titik referensi
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    # Hitung jarak antara ujung jari dan pergelangan tangan
    thumb_dist = calculate_distance(thumb_tip, wrist)
    index_dist = calculate_distance(index_tip, wrist)
    middle_dist = calculate_distance(middle_tip, wrist)
    ring_dist = calculate_distance(ring_tip, wrist)
    pinky_dist = calculate_distance(pinky_tip, wrist)
    
    # Hitung jarak antara ujung jari dan ujung ibu jari
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
    thumb_ring_dist = calculate_distance(thumb_tip, ring_tip)
    thumb_pinky_dist = calculate_distance(thumb_tip, pinky_tip)
    
    # Deteksi gesture
    if (index_dist > 0.2 and middle_dist > 0.2 and 
        ring_dist > 0.2 and pinky_dist > 0.2 and thumb_dist > 0.15):
        return "OPEN HAND"
    elif (index_dist < 0.1 and middle_dist < 0.1 and 
          ring_dist < 0.1 and pinky_dist < 0.1 and thumb_dist > 0.1):
        return "FIST"
    elif (thumb_index_dist < 0.05 and middle_dist > 0.15 and 
          ring_dist > 0.15 and pinky_dist > 0.15):
        return "OK"
    elif (index_dist > 0.2 and middle_dist < 0.1 and 
          ring_dist < 0.1 and pinky_dist < 0.1):
        return "POINTING"
    elif (index_dist > 0.2 and middle_dist > 0.2 and 
          ring_dist < 0.1 and pinky_dist < 0.1):
        return "VICTORY"
    elif (index_dist > 0.2 and pinky_dist > 0.2 and 
          middle_dist < 0.1 and ring_dist < 0.1):
        return "ROCK"
    else:
        return "UNKNOWN"

def detect_gesture(landmarks):
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


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Flip frame secara horizontal untuk mirror effect
    frame = cv2.flip(frame, 1)
    
    # Konversi BGR ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Proses frame dengan MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Dapatkan koordinat landmark
            landmarks = hand_landmarks.landmark
            
            # Deteksi gesture
            gesture = detect_gesture(landmarks)
            
            # Tampilkan gesture di layar
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Tampilkan frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()