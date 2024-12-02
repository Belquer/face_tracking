import cv2
import numpy as np
from pythonosc import udp_client
import time
import dlib

def list_available_cameras():
    """List all available camera devices."""
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

class FaceTracker:
    def __init__(self, camera_index=0, osc_ip="127.0.0.1", osc_port=12345):
        # Initialize OSC client
        self.osc_client = udp_client.SimpleUDPClient(osc_ip, osc_port)
        
        # Initialize webcam
        print(f"Initializing webcam (index: {camera_index})...")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam at index {camera_index}")
        
        # Load Dlib's face detector and shape predictor
        print("Loading detection models...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )
        
        print("Initialization complete. Starting tracking...")

    def detect_features(self, frame, gray):
        # Detect faces using Dlib
        faces = self.detector(gray)
        
        for face in faces:
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            
            # Draw face rectangle
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Calculate mouth opening
            mouth_opening = landmarks.part(66).y - landmarks.part(62).y
            self.send_osc_data("/mouth/opening", mouth_opening)
            
            # Calculate eye opening for both eyes
            left_eye_opening = landmarks.part(41).y - landmarks.part(37).y
            right_eye_opening = landmarks.part(47).y - landmarks.part(43).y
            self.send_osc_data("/eye/left/opening", left_eye_opening)
            self.send_osc_data("/eye/right/opening", right_eye_opening)
            
            # Mouth position (center)
            mouth_x = (landmarks.part(48).x + landmarks.part(54).x) / 2
            mouth_y = (landmarks.part(51).y + landmarks.part(57).y) / 2
            norm_mouth_x = mouth_x / frame.shape[1]
            norm_mouth_y = mouth_y / frame.shape[0]
            self.send_osc_data("/mouth/x", norm_mouth_x)
            self.send_osc_data("/mouth/y", norm_mouth_y)
            
            # Eye positions (center)
            left_eye_x = (landmarks.part(36).x + landmarks.part(39).x) / 2
            left_eye_y = (landmarks.part(37).y + landmarks.part(41).y) / 2
            right_eye_x = (landmarks.part(42).x + landmarks.part(45).x) / 2
            right_eye_y = (landmarks.part(43).y + landmarks.part(47).y) / 2
            norm_left_eye_x = left_eye_x / frame.shape[1]
            norm_left_eye_y = left_eye_y / frame.shape[0]
            norm_right_eye_x = right_eye_x / frame.shape[1]
            norm_right_eye_y = right_eye_y / frame.shape[0]
            self.send_osc_data("/eye/left/x", norm_left_eye_x)
            self.send_osc_data("/eye/left/y", norm_left_eye_y)
            self.send_osc_data("/eye/right/x", norm_right_eye_x)
            self.send_osc_data("/eye/right/y", norm_right_eye_y)
            
            # Draw landmarks
            for n in range(36, 48):  # Eyes
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
            for n in range(48, 68):  # Mouth
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            # Send OSC data for face and mouth
            norm_x = (x + w/2) / frame.shape[1]
            norm_y = (y + h/2) / frame.shape[0]
            self.send_osc_data("/face/x", norm_x)
            self.send_osc_data("/face/y", norm_y)
            self.send_osc_data("/face/size", w * h)
            
            # Only process the largest face
            break

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram to improve detection
        gray = cv2.equalizeHist(gray)
        
        # Detect and draw features
        self.detect_features(frame, gray)

        # Display the frame
        cv2.imshow('Face Tracking', frame)
        return True

    def send_osc_data(self, address, value):
        self.osc_client.send_message(address, float(value))

    def run(self):
        try:
            print("Starting tracking. Press 'q' to quit.")
            while True:
                if not self.process_frame():
                    print("Error reading from webcam")
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break
                
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error during tracking: {e}")
        finally:
            print("Cleaning up...")
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        available_cameras = list_available_cameras()
        print(f"Available cameras: {available_cameras}")
        camera_index = int(input("Enter the index of the camera to use: "))
        if camera_index not in available_cameras:
            raise RuntimeError("Invalid camera index")
        
        tracker = FaceTracker(camera_index)
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
