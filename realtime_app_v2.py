import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms
from PIL import Image
import os
import sys
import time
import math

# --- 1. CẤU HÌNH ĐƯỜNG DẪN IMPORT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)

try:
    from models.texture_cnn import TextureCNN
except ImportError:
    sys.path.append(os.path.dirname(current_dir))
    from models.texture_cnn import TextureCNN

# --- 2. CÁC HÀM TÍNH TOÁN ---
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_ear(landmarks, eye_indices):
    v1 = calculate_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    v2 = calculate_distance(landmarks[eye_indices[3]], landmarks[eye_indices[5]])
    h = calculate_distance(landmarks[eye_indices[0]], landmarks[eye_indices[1]])
    if h == 0: return 0.0
    return (v1 + v2) / (2.0 * h)

def get_mar(landmarks, mouth_indices):
    v = calculate_distance(landmarks[13], landmarks[14]) 
    h = calculate_distance(landmarks[61], landmarks[291])
    if h == 0: return 0.0
    return v / h

def get_smile_ratio(landmarks):
    mouth_w = calculate_distance(landmarks[61], landmarks[291])
    face_w = calculate_distance(landmarks[234], landmarks[454])
    if face_w == 0: return 0.0
    return mouth_w / face_w

# --- 3. CẤU HÌNH THAM SỐ ---
LEFT_EYE_IDXS  = [33, 133, 160, 158, 144, 153]
RIGHT_EYE_IDXS = [362, 263, 385, 387, 373, 380]
MOUTH_IDXS     = [61, 291, 13, 14]

EYE_AR_THRESH = 0.22      
EYE_AR_CONSEC_FRAMES = 2
MOUTH_AR_THRESH = 0.5     
SMILE_RATIO_THRESH = 0.45 


LABELS = ['Real', 'Spoof'] # Nếu bạn là người thật mà bị báo đỏ, HÃY ĐỔI VỊ TRÍ 2 CHỮ NÀY

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model_path = os.path.join(current_dir, 'models', 'trained_model.pth')
    if not os.path.exists(model_path):
        print("[LỖI] Chưa có file model!")
        return

    texture_model = TextureCNN(num_classes=2).to(DEVICE)
    texture_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    texture_model.eval()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cap = cv2.VideoCapture(0)
    
    # Biến đếm
    TOTAL_BLINKS = 0
    TOTAL_SMILES = 0
    TOTAL_MOUTH_OPENS = 0

    BLINK_COUNTER = 0
    IS_MOUTH_OPEN = False
    IS_SMILING = False

    STATE = 0 
    IS_REAL_TEXTURE = False

    print("ĐÃ MỞ KHÓA HIỂN THỊ - Bấm 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)

        # Mặc định vẽ bảng thông số (Vẽ trước để không bị che)
        # Bảng đen góc trái
        cv2.rectangle(frame, (0, 0), (250, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Blinks: {TOTAL_BLINKS}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Smiles: {TOTAL_SMILES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Mouth Ops: {TOTAL_MOUTH_OPENS}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark
                
                # Tìm tọa độ mặt
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for pt in lm:
                    x, y = int(pt.x * w), int(pt.y * h)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)
                
                x_min = max(0, x_min - 20)
                y_min = max(0, y_min - 20)
                x_max = min(w, x_max + 20)
                y_max = min(h, y_max + 20)

                # --- 1. CHECK TEXTURE ---
                # Chỉ check nếu chưa pass hết các bước
                if STATE < 3:
                    try:
                        face_roi = frame[y_min:y_max, x_min:x_max]
                        if face_roi.size > 0:
                            pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                            input_tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                output = texture_model(input_tensor)
                                probs = torch.nn.functional.softmax(output, dim=1)
                                _, preds = torch.max(probs, 1)
                            
                            label = LABELS[preds.item()]
                            
                            if label == 'Real':
                                IS_REAL_TEXTURE = True
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                cv2.putText(frame, "REAL (DA THAT)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            else:
                                IS_REAL_TEXTURE = False
                                STATE = 0 # Reset game nếu mất texture thật
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                                cv2.putText(frame, "FAKE (GIA MAO)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    except: pass

                # --- 2. TÍNH TOÁN HÀNH VI (CHẠY LUÔN DÙ LÀ REAL HAY FAKE) ---
                # (Đã bỏ đoạn 'if not IS_REAL_TEXTURE: continue' để luôn hiện thông số)
                left_ear = get_ear(lm, LEFT_EYE_IDXS)
                right_ear = get_ear(lm, RIGHT_EYE_IDXS)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = get_mar(lm, MOUTH_IDXS)
                smile_ratio = get_smile_ratio(lm)

                # --- A. ĐẾM CHỚP MẮT ---
                if avg_ear < EYE_AR_THRESH:
                    BLINK_COUNTER += 1
                else:
                    if BLINK_COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL_BLINKS += 1
                    BLINK_COUNTER = 0

                # --- B. ĐẾM MỞ MIỆNG ---
                if mar > MOUTH_AR_THRESH:
                    if not IS_MOUTH_OPEN:
                        TOTAL_MOUTH_OPENS += 1
                        IS_MOUTH_OPEN = True
                    cv2.putText(frame, "MO MIENG!", (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    IS_MOUTH_OPEN = False

                # --- C. ĐẾM CƯỜI ---
                if smile_ratio > SMILE_RATIO_THRESH and mar < 0.3:
                    if not IS_SMILING:
                        TOTAL_SMILES += 1
                        IS_SMILING = True
                    cv2.putText(frame, "DANG CUOI!", (x_min, y_max + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    IS_SMILING = False

                # --- 3. QUY TRÌNH GAME (Chỉ chạy khi Texture = Real) ---
                if IS_REAL_TEXTURE:
                    if STATE == 0: 
                        STATE = 1
                    
                    elif STATE == 1: # Chớp mắt
                        cv2.putText(frame, f"BUOC 1: CHOP MAT ({TOTAL_BLINKS}/1)", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if TOTAL_BLINKS >= 1: STATE = 2
                    
                    elif STATE == 2: # Cười/Nói
                        cv2.putText(frame, "BUOC 2: CUOI HOAC NOI", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if TOTAL_SMILES >= 1 or TOTAL_MOUTH_OPENS >= 1: STATE = 3
                    
                    elif STATE == 3:
                        cv2.putText(frame, "XAC THUC THANH CONG!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.imshow('Face Liveness', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()