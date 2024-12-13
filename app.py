#
# import cv2
# import numpy as np
#
# import joblib
#
# def load_model_and_scaler(model_path="svm_model.pkl", scaler_path="scaler.pkl"):
#     # Load the trained model
#     model = joblib.load(model_path)
#     print(f"Model loaded from {model_path}")
#
#     # Load the scaler
#     scaler = joblib.load(scaler_path)
#     print(f"Scaler loaded from {scaler_path}")
#
#     return model, scaler
#
#
# # Load the model and scaler
# loaded_model, loaded_scaler = load_model_and_scaler("svm_model.pkl", "scaler.pkl")
#
#
# def extract_features(image, bbox):
#     xmin, ymin, xmax, ymax = bbox
#     roi = image[ymin:ymax, xmin:xmax]
#
#     # Debugging: Print the ROI dimensions
#     # print(f"Bounding Box: {bbox}")
#     # print(f"ROI shape before resizing: {roi.shape}")
#
#     # Ensure the ROI has valid dimensions
#     if roi.size == 0:
#         print("Invalid ROI: Skipping...")
#         return None  # Return None for invalid ROIs
#
#     # Resize ROI to a fixed size (e.g., 64x128 for HOG compatibility)
#     try:
#         roi_resized = cv2.resize(roi, (64, 128))
#     except Exception as e:
#         print(f"Error resizing ROI: {e}")
#         return None
#
#     # Convert ROI to grayscale (required by HOG)
#     roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)
#
#     # HOG features
#     hog = cv2.HOGDescriptor()
#     hog_features = hog.compute(roi_gray).flatten()
#
#     # Edge detection (Canny)
#     edges = cv2.Canny(roi_gray, 100, 200)
#     edge_features = edges.flatten()
#
#     # Color histogram
#     hist = cv2.calcHist([roi_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     color_features = hist.flatten()
#
#     # Combine all features
#     return np.hstack((hog_features, edge_features, color_features))
# # Function to process live video feed for knife detection
# def detect_knife_live_video(model, scaler, window_size=64, step_size=32):
#     # Open the default webcam
#     cap = cv2.VideoCapture(0)
#
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
#
#     while True:
#         # Read a frame from the webcam
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break
#
#         # Resize frame for faster processing
#         frame = cv2.resize(frame, (640, 480))
#         h, w, _ = frame.shape
#
#         # Detect knives using a sliding window
#         detected_boxes = []
#         for y in range(0, h - window_size, step_size):
#             for x in range(0, w - window_size, step_size):
#                 roi = frame[y:y + window_size, x:x + window_size]
#                 roi_features = extract_features(frame, (x, y, x + window_size, y + window_size))
#                 roi_features = scaler.transform([roi_features])
#                 prediction = model.predict(roi_features)
#
#                 if prediction == 1:  # Knife detected
#                     detected_boxes.append((x, y, x + window_size, y + window_size))
#
#         # Draw bounding boxes on the frame
#         for (xmin, ymin, xmax, ymax) in detected_boxes:
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(frame, "Knife Detected", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#         # Display the frame
#         cv2.imshow("Knife Detection", frame)
#
#         # Break loop on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release the webcam and close windows
#     cap.release()
#     cv2.destroyAllWindows()
#
# # Main script
# if __name__ == "__main__":
#     # Paths for saved model and scaler
#     model_path = "svm_model.pkl"
#     scaler_path = "scaler.pkl"
#
#     # Load the trained model and scaler
#     model, scaler = load_model_and_scaler(model_path, scaler_path)
#
#     # Start live video knife detection
#     detect_knife_live_video(model, scaler)
#



import cv2
import numpy as np
import joblib
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the model and scaler
def load_model_and_scaler(model_path="svm_model.pkl", scaler_path="scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Feature extraction
def extract_features(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    roi = image[ymin:ymax, xmin:xmax]

    if roi.size == 0:
        return None

    try:
        roi_resized = cv2.resize(roi, (64, 128))
    except Exception as e:
        return None

    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)

    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(roi_gray).flatten()

    edges = cv2.Canny(roi_gray, 100, 200)
    edge_features = edges.flatten()

    hist = cv2.calcHist([roi_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_features = hist.flatten()

    return np.hstack((hog_features, edge_features, color_features))

# Transformer for Streamlit WebRTC
class KnifeDetectionTransformer(VideoTransformerBase):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape
        window_size = 64
        step_size = 32

        detected_boxes = []
        for y in range(0, h - window_size, step_size):
            for x in range(0, w - window_size, step_size):
                roi_features = extract_features(image, (x, y, x + window_size, y + window_size))
                if roi_features is None:
                    continue
                roi_features = self.scaler.transform([roi_features])
                prediction = self.model.predict(roi_features)

                if prediction == 1:  # Knife detected
                    detected_boxes.append((x, y, x + window_size, y + window_size))

        for (xmin, ymin, xmax, ymax) in detected_boxes:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, "Knife Detected", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return image

# Streamlit App
st.title("Live Knife Detection Web App")
st.write("This application uses a trained SVM model to detect knives in a live video feed.")

# Load model and scaler
model_path = "svm_model.pkl"
scaler_path = "scaler.pkl"
st.write("Loading model and scaler...")

try:
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Start the video stream
webrtc_streamer(
    key="knife-detection",
    video_transformer_factory=lambda: KnifeDetectionTransformer(model, scaler),
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)
