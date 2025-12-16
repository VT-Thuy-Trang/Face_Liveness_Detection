import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            # Trả về landmarks của khuôn mặt đầu tiên
            return results.multi_face_landmarks[0], image.shape
        return None, None
    
    def get_bbox(self, landmarks, frame_shape):
        h, w, _ = frame_shape
        pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])
        x, y, w_rect, h_rect = cv2.boundingRect(pts)
        return x, y, w_rect, h_rect