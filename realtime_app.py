import cv2
import numpy as np
import time
import random
from detectors.face_detector import FaceDetector
from detectors.emotion_detector import EmotionDetector
from detectors.motion_detector import MotionDetector

STATE_WAITING = 0      # Chờ khuôn mặt ổn định
STATE_ANALYZING = 1    # Phân tích ảnh tĩnh/động
STATE_CHALLENGE = 2    # Thử thách hành động
STATE_RESULT = 3       # Hiển thị kết quả

# Ngưỡng phát hiện ảnh tĩnh (Nếu motion_score < 1.5 -> Ảnh)
STATIC_THRESHOLD = 1.5 

def main():
    # 1. Khởi tạo
    face_det = FaceDetector()
    emotion_det = EmotionDetector()
    motion_det = MotionDetector() # Class mới từ file riêng

    cap = cv2.VideoCapture(0)
    
    # Biến trạng thái
    current_state = STATE_WAITING
    
    # Biến Challenge
    challenge_type = ""
    challenge_timer = 0
    CHALLENGE_LIMIT = 5.0
    
    # Biến Kết quả
    result_text = ""
    result_color = (0,0,0)
    result_timer = 0

    print("--- Hệ thống liveness---")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 2. Phát hiện mặt
        landmarks, shape = face_det.detect(frame) 
        
        if not landmarks:
            # Reset khi không có mặt
            current_state = STATE_WAITING # Trở về trạng thái chờ
            motion_det.reset()
            
            cv2.putText(frame, "Waiting for face...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2) # Hiển thị thông báo
        
        else:
            # Lấy tọa độ khung mặt
            fx, fy, fw, fh = face_det.get_bbox(landmarks, shape)
        
            # Vẽ khung 
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            #hiển thị dù đang ở trạng thái nào
            current_emotion, emotion_color = emotion_det.detect_state(landmarks, w, h)
            
            # Hiển thị text cảm xúc ngay trên đầu khung mặt
            cv2.putText(frame, f"Emotion: {current_emotion}", (fx, fy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)

            # Cập nhật điểm chuyển động
            motion_score = motion_det.update(landmarks)
            
            #Hiển thị chỉ số Motion ở dưới đáy khung
            cv2.putText(frame, f"Motion Score: {motion_score:.2f}", (fx, fy + fh + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            
            if current_state == STATE_WAITING:
                # Chờ thu thập đủ dữ liệu chuyển động (khoảng 20 frames)
                if len(motion_det.history) >= 20:
                    current_state = STATE_ANALYZING

            elif current_state == STATE_ANALYZING:
                cv2.putText(frame, "Checking Static...", (fx, fy - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Kiểm tra ảnh tĩnh
                if motion_score < STATIC_THRESHOLD:
                    current_state = STATE_RESULT
                    result_text = "FAKE: STATIC PHOTO"
                    result_color = (0, 0, 255) # Đỏ
                    result_timer = time.time()
                else:
                    # Nếu không phải ảnh tĩnh -> Vào thử thách
                    challenges = ["SMILE", "SURPRISE", "BLINK"]
                    challenge_type = random.choice(challenges)
                    challenge_timer = time.time()
                    current_state = STATE_CHALLENGE

            elif current_state == STATE_CHALLENGE:
                elapsed = time.time() - challenge_timer
                time_left = CHALLENGE_LIMIT - elapsed
                
                # Hiển thị yêu cầu
                msg = f"Challenge: {challenge_type}"
                cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"Time: {time_left:.1f}s", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)

                # Bảo vệ: Vẫn check ảnh tĩnh liên tục trong lúc challenge
                if motion_score < STATIC_THRESHOLD:
                     current_state = STATE_RESULT
                     result_text = "FAKE: STATIC DETECTED"
                     result_color = (0, 0, 255)
                     result_timer = time.time()

                # Kiểm tra cảm xúc có khớp với yêu cầu không
                passed = False
                if challenge_type == "SMILE" and current_emotion == "SMILING": passed = True
                elif challenge_type == "SURPRISE" and current_emotion == "SURPRISED": passed = True
                elif challenge_type == "BLINK" and "BLINKING" in current_emotion: passed = True

                if passed: # Thử thách thành công
                    current_state = STATE_RESULT
                    result_text = "ACCESS GRANTED"
                    result_color = (0, 255, 0) # Xanh lá
                    result_timer = time.time()
                
                if time_left <= 0: # Hết thời gian thử thách
                    current_state = STATE_RESULT
                    result_text = "FAILED: TIME OUT"
                    result_color = (0, 0, 255) 
                    result_timer = time.time()

            elif current_state == STATE_RESULT: # Hiển thị kết quả
                # Hiển thị kết quả to giữa màn hình
                cv2.putText(frame, result_text, (50, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 4)
                
                # Tự động reset sau 3 giây
                if time.time() - result_timer > 3.0:
                    current_state = STATE_WAITING
                    motion_det.reset() # Xóa lịch sử chuyển động cũ

        cv2.imshow("Face Liveness System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()