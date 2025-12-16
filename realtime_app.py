import cv2
import numpy as np
from detectors.face_detector import FaceDetector
from detectors.texture_detector import TextureDetector
from detectors.blink_detector import BlinkDetector

def main():
    # 1. Khởi tạo
    face_det = FaceDetector()
    tex_det = TextureDetector(model_path='models/trained_model.pth')
    blink_det = BlinkDetector(threshold=0.25)

    cap = cv2.VideoCapture(0)
    
    blink_count = 0
    blink_registered = False # Cờ đánh dấu trạng thái mắt



    while True:
        ret, frame = cap.read()
        if not ret: break

        # 2. Phát hiện mặt
        landmarks, shape = face_det.detect(frame)
        
        status = "Waiting..."
        color = (200, 200, 200)

        if landmarks:
            h, w, _ = shape
            x, y, w_rect, h_rect = face_det.get_bbox(landmarks, shape)
            
            # Mở rộng vùng mặt để lấy texture tốt hơn
            pad = 10
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(w, x+w_rect+pad), min(h, y+h_rect+pad)
            face_crop = frame[y1:y2, x1:x2]

            # 3. Check Texture (Deep Learning)
            if face_crop.size > 0:
                real_score = tex_det.predict(face_crop)
                
                # 4. Check Blink (Logic)
                is_closed = blink_det.check(landmarks, w, h)
                if is_closed:
                    blink_registered = True
                
                if blink_registered and not is_closed:
                    blink_count += 1
                    blink_registered = False # Reset sau khi mở mắt ra

                # 5. Quyết định
                # Điều kiện: Texture phải giống thật (>70%) VÀ đã chớp mắt ít nhất 1 lần
                if real_score > 0.7:
                    if blink_count >= 1:
                        status = "REAL FACE (ACCESS GRANTED)"
                        color = (0, 255, 0) # Xanh lá
                    else:
                        status = "Texture OK. Please Blink!"
                        color = (0, 255, 255) # Vàng
                else:
                    status = "FAKE / SPOOF DETECTED"
                    color = (0, 0, 255) # Đỏ

            # Vẽ khung
            cv2.rectangle(frame, (x, y), (x+w_rect, y+h_rect), color, 2)
            cv2.putText(frame, f"Score: {real_score:.2f} | Blinks: {blink_count}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Hiển thị
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Project Demo - VS2022", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()