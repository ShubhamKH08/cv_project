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

# -------------- initial streamlit deployed code

# import cv2
# import numpy as np
# import joblib
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
#
# # Load the model and scaler
# def load_model_and_scaler(model_path="svm_model.pkl", scaler_path="scaler.pkl"):
#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)
#     return model, scaler
#
# # Feature extraction
# def extract_features(image, bbox):
#     xmin, ymin, xmax, ymax = bbox
#     roi = image[ymin:ymax, xmin:xmax]
#
#     if roi.size == 0:
#         return None
#
#     try:
#         roi_resized = cv2.resize(roi, (64, 128))
#     except Exception as e:
#         return None
#
#     roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)
#
#     hog = cv2.HOGDescriptor()
#     hog_features = hog.compute(roi_gray).flatten()
#
#     edges = cv2.Canny(roi_gray, 100, 200)
#     edge_features = edges.flatten()
#
#     hist = cv2.calcHist([roi_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     color_features = hist.flatten()
#
#     return np.hstack((hog_features, edge_features, color_features))
#
# # Transformer for Streamlit WebRTC
# class KnifeDetectionTransformer(VideoTransformerBase):
#     def __init__(self, model, scaler):
#         self.model = model
#         self.scaler = scaler
#
#
#     def transform(self, frame):
#         image = frame.to_ndarray(format="bgr24")
#         print("Frame received for processing")  # Debugging
#
#         h, w, _ = image.shape
#         window_size = 64
#         step_size = 32
#
#         detected_boxes = []
#         for y in range(0, h - window_size, step_size):
#             for x in range(0, w - window_size, step_size):
#                 roi_features = extract_features(image, (x, y, x + window_size, y + window_size))
#                 if roi_features is None:
#                     print(f"Skipping invalid ROI at: {x}, {y}")  # Debugging
#                     continue
#                 roi_features = self.scaler.transform([roi_features])
#                 prediction = self.model.predict(roi_features)
#
#                 if prediction == 1:  # Knife detected
#                     print(f"Knife detected at: {x}, {y}")  # Debugging
#                     detected_boxes.append((x, y, x + window_size, y + window_size))
#
#         for (xmin, ymin, xmax, ymax) in detected_boxes:
#             cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(image, "Knife Detected", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#         print(f"Processed frame with {len(detected_boxes)} detections")  # Debugging
#         return image
#
#
# # Streamlit App
# st.title("Live Knife Detection Web App")
# st.write("This application uses a trained SVM model to detect knives in a live video feed.")
#
# # Load model and scaler
# model_path = "svm_model.pkl"
# scaler_path = "scaler.pkl"
# st.write("Loading model and scaler...")
#
# try:
#     model, scaler = load_model_and_scaler(model_path, scaler_path)
#     st.success("Model and scaler loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading model or scaler: {e}")
#     st.stop()
#
# # Start the video stream
# webrtc_streamer(
#     key="knife-detection",
#     video_transformer_factory=lambda: KnifeDetectionTransformer(model, scaler),
#     media_stream_constraints={
#         "video": True,
#         "audio": False
#     },
# )


# --- working code is below

# import cv2
# import numpy as np
# import joblib
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# # Load the model and scaler
# def load_model_and_scaler(model_path="svm_model.pkl", scaler_path="scaler.pkl"):
#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)
#     return model, scaler

# # Feature extraction
# def extract_features(image, bbox):
#     xmin, ymin, xmax, ymax = bbox
#     roi = image[ymin:ymax, xmin:xmax]

#     if roi.size == 0:
#         return None

#     try:
#         roi_resized = cv2.resize(roi, (64, 128))
#     except Exception as e:
#         return None

#     roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)

#     hog = cv2.HOGDescriptor()
#     hog_features = hog.compute(roi_gray).flatten()

#     edges = cv2.Canny(roi_gray, 100, 200)
#     edge_features = edges.flatten()

#     hist = cv2.calcHist([roi_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     color_features = hist.flatten()

#     return np.hstack((hog_features, edge_features, color_features))

# # Transformer for Streamlit WebRTC
# class KnifeDetectionTransformer(VideoTransformerBase):
#     def __init__(self, model, scaler):
#         self.model = model
#         self.scaler = scaler
#         self.frame_count = 0  # Count frames to sample FPS

#     def transform(self, frame):
#         self.frame_count += 1
#         sample_rate = 6  # Process every 6th frame for ~5 FPS

#         # Skip frames to reduce processing frequency
#         if self.frame_count % sample_rate != 0:
#             return frame.to_ndarray(format="bgr24")

#         image = frame.to_ndarray(format="bgr24")
#         h, w, _ = image.shape
#         window_size = 64
#         step_size = 32

#         detected_boxes = []
#         for y in range(0, h - window_size, step_size):
#             for x in range(0, w - window_size, step_size):
#                 roi_features = extract_features(image, (x, y, x + window_size, y + window_size))
#                 if roi_features is None:
#                     continue
#                 roi_features = self.scaler.transform([roi_features])
#                 prediction = self.model.predict(roi_features)

#                 if prediction == 1:  # Knife detected
#                     detected_boxes.append((x, y, x + window_size, y + window_size))

#         for (xmin, ymin, xmax, ymax) in detected_boxes:
#             cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(image, "Knife Detected", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         return image

# # Streamlit App
# st.title("Live Knife Detection Web App")
# st.write("This application uses a trained SVM model to detect knives in a live video feed.")

# # Load model and scaler
# model_path = "svm_model.pkl"
# scaler_path = "scaler.pkl"
# st.write("Loading model and scaler...")

# try:
#     model, scaler = load_model_and_scaler(model_path, scaler_path)
#     st.success("Model and scaler loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading model or scaler: {e}")
#     st.stop()

# # Start the video stream
# # webrtc_streamer(
# #     key="knife-detection",
# #     video_transformer_factory=lambda: KnifeDetectionTransformer(model, scaler),
# #     media_stream_constraints={
# #         "video": True,
# #         "audio": False
# #     },
# # )


# from streamlit_webrtc import RTCConfiguration

# rtc_config = RTCConfiguration({
#     "iceServers": [
#         {"urls": ["stun:stun.l.google.com:19302"]},  # Google's public STUN server
#         {
#             "urls": ["turn:relay.metered.ca:443"],
#             "username": "open",
#             "credential": "open"
#         },
#     ]
# })

# webrtc_streamer(
#     key="knife-detection",
#     video_transformer_factory=lambda: KnifeDetectionTransformer(model, scaler),
#     rtc_configuration=rtc_config,  # Pass the RTCConfiguration
#     media_stream_constraints={
#         "video": True,
#         "audio": False
#     },
# )


## cartoon code 


# import cv2
# import streamlit as st
# import numpy as np

# # Streamlit app title and description
# st.title("Real-Time Cartoon Filter")
# st.write("A cartoon filter with enhancements and fun face filters. Click 'Start Video' to begin and explore features using the buttons.")

# # Define Start and Stop buttons
# start_button = st.button("Start Video")
# stop_button = st.button("Stop Video")

# # Feature toggle buttons
# cartoon_enhance_button = st.button("Cartoon Enhance")
# emoji_mask_button = st.button("Emoji Masks")
# custom_accessories_button = st.button("Custom Accessories")

# # Initialize session state variables
# if "run" not in st.session_state:
#     st.session_state.run = False
# if "feature" not in st.session_state:
#     st.session_state.feature = None

# # Update session state based on button clicks
# if start_button:
#     st.session_state.run = True
# if stop_button:
#     st.session_state.run = False
# if cartoon_enhance_button:
#     st.session_state.feature = "enhance"
# if emoji_mask_button:
#     st.session_state.feature = "emoji"
# if custom_accessories_button:
#     st.session_state.feature = "accessories"

# # Cartoonization parameters
# BILATERAL_FILTER_VALUE = 5  # Reduced for better speed
# COLOR_QUANTIZATION_LEVEL = 8  # Reduced for faster processing

# # Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load overlay images for accessories
# hat_img = cv2.imread("hat.png", -1)  # Ensure you have this file
# sunglasses_img = cv2.imread("sunglasses.png", -1)  # Ensure you have this file

# def apply_bilateral_filter(frame):
#     """Smooths the image while preserving edges using bilateral filtering."""
#     return cv2.bilateralFilter(frame, BILATERAL_FILTER_VALUE, 75, 75)

# def color_quantization(frame, k=COLOR_QUANTIZATION_LEVEL):
#     """Applies color quantization to reduce the color palette of the image."""
#     data = np.float32(frame).reshape((-1, 3))
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     _, labels, palette = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
#     quantized = palette[labels.flatten()].reshape(frame.shape)
#     return quantized.astype(np.uint8)

# def detect_edges_stylized(gray_frame):
#     """Detects edges using a stylized filter approach."""
#     edges = cv2.adaptiveThreshold(
#         gray_frame, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         blockSize=9,
#         C=2
#     )
#     return edges

# def overlay_image(background, overlay, x, y):
#     """Overlays a transparent image onto a background image."""
#     for c in range(0, 3):
#         alpha = overlay[:, :, 3] / 255.0
#         background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] = (
#             alpha * overlay[:, :, c] +
#             (1 - alpha) * background[y:y+overlay.shape[0], x:x+overlay.shape[1], c]
#         )

# def cartoonize_frame(frame):
#     """Main cartoonization pipeline."""
#     filtered = apply_bilateral_filter(frame)
#     gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
#     edges = detect_edges_stylized(gray)
#     quantized = color_quantization(filtered)
#     edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     cartoon = cv2.bitwise_and(quantized, edges_colored)
#     return cartoon

# def enhance_cartoon(frame):
#     """Applies enhancements to the cartoon effect."""
#     cartoon = cartoonize_frame(frame)
#     return cv2.applyColorMap(cartoon, cv2.COLORMAP_HOT)

# def apply_emoji_mask(frame):
#     """Detects faces and applies emoji masks."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#     emoji = cv2.imread("emoji.png", -1)  # Ensure you have this file

#     for (x, y, w, h) in faces:
#         emoji_resized = cv2.resize(emoji, (w, h))
#         overlay_image(frame, emoji_resized, x, y)
#     return frame

# def apply_custom_accessories(frame):
#     """Detects faces and applies custom accessories like hats and sunglasses."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#     for (x, y, w, h) in faces:
#         # Add hat
#         hat_resized = cv2.resize(hat_img, (w, int(0.5 * h)))
#         overlay_image(frame, hat_resized, x, y - int(0.5 * h))

#         # Add sunglasses
#         sunglasses_resized = cv2.resize(sunglasses_img, (w, int(0.3 * h)))
#         overlay_image(frame, sunglasses_resized, x, y + int(0.2 * h))
#     return frame

# # Open video capture if Start button is clicked
# cap = cv2.VideoCapture(0)

# if st.session_state.run:
#     stframe = st.empty()  # Placeholder for video frames

#     while st.session_state.run:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("Unable to access webcam.")
#             break

#         if st.session_state.feature == "enhance":
#             frame = enhance_cartoon(frame)
#         elif st.session_state.feature == "emoji":
#             frame = apply_emoji_mask(frame)
#         elif st.session_state.feature == "accessories":
#             frame = apply_custom_accessories(frame)
#         else:
#             frame = cartoonize_frame(frame)

#         # Display the processed video feed
#         stframe.image(frame, channels="BGR")

# # Release video capture when Stop button is clicked
# if not st.session_state.run and cap.isOpened():
#     cap.release()
#     st.write("Video stopped.")

## error in code st
# import cv2
# import streamlit as st
# import numpy as np

# # Updated Model Loading and Detection Logic
# import joblib

# def load_model_and_scaler(model_path="svm_model.pkl", scaler_path="scaler.pkl"):
#     """Load the pre-trained model and scaler for knife detection."""
#     try:
#         model = joblib.load(model_path)
#         scaler = joblib.load(scaler_path)
#         st.success("Model and scaler loaded successfully!")
#         return model, scaler
#     except Exception as e:
#         st.error(f"Error loading model or scaler: {e}")
#         return None, None

# model_path = "svm_model.pkl"
# scaler_path = "scaler.pkl"
# knife_model, knife_scaler = load_model_and_scaler(model_path, scaler_path)

# def detect_knives_with_model(image, model, scaler):
#     """Applies knife detection using the loaded model and scaler."""
#     if model is None or scaler is None:
#         st.error("Knife detection model or scaler is not loaded!")
#         return image

#     h, w, _ = image.shape
#     window_size = 64
#     step_size = 32
#     detected_boxes = []

#     for y in range(0, h - window_size, step_size):
#         for x in range(0, w - window_size, step_size):
#             roi = image[y:y + window_size, x:x + window_size]
#             if roi.size == 0:
#                 continue

#             try:
#                 roi_resized = cv2.resize(roi, (64, 128))
#                 roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
#                 hog = cv2.HOGDescriptor()
#                 hog_features = hog.compute(roi_gray).flatten()
#                 edges = cv2.Canny(roi_gray, 100, 200).flatten()
#                 hist = cv2.calcHist([roi_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
#                 features = np.hstack((hog_features, edges, hist))
#                 features = scaler.transform([features])
#                 prediction = model.predict(features)
#                 if prediction == 1:  # Knife detected
#                     detected_boxes.append((x, y, x + window_size, y + window_size))
#             except Exception:
#                 continue

#     for (xmin, ymin, xmax, ymax) in detected_boxes:
#         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#         cv2.putText(image, "Knife Detected", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     return image

# # Streamlit app title and description
# st.title("Real-Time Cartoon Filter and Knife Detection")
# st.write("Choose between real-time video processing or image upload for knife detection.")

# # Define mode selection
# mode = st.radio("Select Mode", ("Video Feed", "Upload Image"))


# def detect_knives(image):
#     """Applies knife detection on the uploaded image and returns the image with bounding boxes."""
#     # Example placeholder detection logic (replace with actual model inference)
#     # Use pre-trained knife detection model to identify knives and draw bounding boxes
#     # For demonstration, we add a mock bounding box
#     height, width = image.shape[:2]
#     start_point = (int(0.3 * width), int(0.3 * height))
#     end_point = (int(0.7 * width), int(0.7 * height))
#     color = (0, 255, 0)  # Green
#     thickness = 2
#     cv2.rectangle(image, start_point, end_point, color, thickness)
#     return image

# if mode == "Video Feed":
#     # Define Start and Stop buttons
#     start_button = st.button("Start Video")
#     stop_button = st.button("Stop Video")

#     # Initialize session state variables
#     if "run" not in st.session_state:
#         st.session_state.run = False

#     if start_button:
#         st.session_state.run = True
#     if stop_button:
#         st.session_state.run = False

#     # Open video capture if Start button is clicked
#     cap = cv2.VideoCapture(0)

#     if st.session_state.run:
#     stframe = st.empty()  # Placeholder for video frames

#     while st.session_state.run:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("Unable to access webcam.")
#             break

#         # Apply knife detection on video frames
#         frame = detect_knives_with_model(frame, knife_model, knife_scaler)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit display
#         stframe.image(frame, channels="RGB")


#     # Release video capture when Stop button is clicked
#     if not st.session_state.run and cap.isOpened():
#         cap.release()
#         st.write("Video stopped.")

# elif mode == "Upload Image":
#     uploaded_file = st.file_uploader("Upload an image for knife detection", type=["jpg", "png", "jpeg"])

#     if uploaded_file is not None:
#     # Read the uploaded image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, 1)

#     # Detect knives using the model
#     output_image = detect_knives_with_model(image, knife_model, knife_scaler)

#     # Convert to RGB for displaying in Streamlit
#     output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

#     st.image(output_image, caption="Knife Detection Result", use_column_width=True)


# st.write("Select a mode above to get started!")

# import streamlit as st
# import cv2
# import numpy as np

# st.title("Webcam Live Feed")

# picture = st.camera_input("Take a picture")

# if picture:
#     # Convert the uploaded image to OpenCV format
#     bytes_data = picture.getvalue()
#     np_arr = np.frombuffer(bytes_data, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     # Display the frame in RGB format
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     st.image(frame, caption="Captured Image", use_container_width=True)

import cv2
import numpy as np
import joblib
import streamlit as st
from PIL import Image

def load_model_and_scaler(model_path="svm_model.pkl", scaler_path="scaler.pkl"):
    # Load the trained model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Load the scaler
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")

    return model, scaler

# Load the model and scaler
loaded_model, loaded_scaler = load_model_and_scaler("svm_model.pkl", "scaler.pkl")

def extract_features(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    roi = image[ymin:ymax, xmin:xmax]

    # Ensure the ROI has valid dimensions
    if roi.size == 0:
        print("Invalid ROI: Skipping...")
        return None

    # Resize ROI to a fixed size (e.g., 64x128 for HOG compatibility)
    try:
        roi_resized = cv2.resize(roi, (64, 128))
    except Exception as e:
        print(f"Error resizing ROI: {e}")
        return None

    # Convert ROI to grayscale (required by HOG)
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    # HOG features
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(roi_gray).flatten()

    # Edge detection (Canny)
    edges = cv2.Canny(roi_gray, 100, 200)
    edge_features = edges.flatten()

    # Color histogram
    hist = cv2.calcHist([roi_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_features = hist.flatten()

    # Combine all features
    return np.hstack((hog_features, edge_features, color_features))

# Streamlit App
st.title("Knife Detection Using SVM")
st.markdown("Live webcam video processing for knife detection using SVM.")

# Access webcam via OpenCV
FRAME_WINDOW = st.image([])

# Open webcam
camera = cv2.VideoCapture(0)

# Check if webcam is opened
if not camera.isOpened():
    st.error("Error: Could not access the webcam.")
else:
    run = st.checkbox("Run Video Stream")

    while run:
        # Capture a frame
        ret, frame = camera.read()
        if not ret:
            st.error("Error: Unable to read from webcam.")
            break

        # Resize frame for processing
        frame_resized = cv2.resize(frame, (640, 480))
        h, w, _ = frame_resized.shape

        # Sliding window parameters
        window_size = 64
        step_size = 32

        # Detect knives using the SVM model
        detected_boxes = []
        for y in range(0, h - window_size, step_size):
            for x in range(0, w - window_size, step_size):
                roi_features = extract_features(
                    frame_resized, (x, y, x + window_size, y + window_size)
                )

                if roi_features is None:
                    continue

                # Scale features and make predictions
                roi_features_scaled = loaded_scaler.transform([roi_features])
                prediction = loaded_model.predict(roi_features_scaled)

                if prediction == 1:  # Knife detected
                    detected_boxes.append((x, y, x + window_size, y + window_size))

        # Draw bounding boxes on the frame
        for (xmin, ymin, xmax, ymax) in detected_boxes:
            cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                frame_resized,
                "Knife Detected",
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        FRAME_WINDOW.image(frame_rgb)

    # Release the webcam when done
    camera.release()
    st.write("Video stream stopped.")

