from scipy.spatial import distance as dist

class BlinkDetector:
    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def _eye_aspect_ratio(self, eye):
        # eye là list các điểm tọa độ (x, y)
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def check(self, landmarks, w, h):
        # Lấy tọa độ mắt trái/phải từ landmarks MediaPipe
        # Index mắt trái: 362, 385, 387, 263, 373, 380
        # Index mắt phải: 33, 160, 158, 133, 153, 144
        
        def get_coords(indices):
            return [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]

        left_eye = get_coords([362, 385, 387, 263, 373, 380])
        right_eye = get_coords([33, 160, 158, 133, 153, 144])

        ear_left = self._eye_aspect_ratio(left_eye)
        ear_right = self._eye_aspect_ratio(right_eye)
        avg_ear = (ear_left + ear_right) / 2.0

        return avg_ear < self.threshold