# Import necessary libraries
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.models import model_from_json
from audio_player import play_audio

# Suppress unnecessary TensorFlow logs
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Load the pre-trained model structure and weights
def load_emotion_model(json_file_path, weights_file_path):
    with open(json_file_path, "r") as json_file:
        model_json = json_file.read()

    # Custom objects in case of deserialization issues
    custom_objects = {"Sequential": Sequential}
    model = model_from_json(model_json, custom_objects=custom_objects)
    model.load_weights(weights_file_path)
    return model


emotion_model = load_emotion_model("facialemotionmodel.json", "facialemotionmodel.h5")

# Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


# Preprocessing function for the neural network
def extract_features(image: np.ndarray) -> np.ndarray:
    feature = np.array(image, dtype="float32").reshape(1, 48, 48, 1)
    return feature / 255.0  # Normalize to 0-1 range


# Emotion labels mapping index -> emotion
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Start capturing video
vid = cv2.VideoCapture(0)

print("Press 'q' to quit the video stream.")
try:
    while True:
        # Read a single video frame
        ret, frame = vid.read()
        if not ret:
            print("Error accessing camera or camera disconnected.")
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using the Haar cascade
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        emotion = ""

        for (x, y, w, h) in faces:
            # Extract and prepare the face ROI for emotion detection
            face_roi = gray_frame[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))

            features = extract_features(face_roi)
            prediction = emotion_model.predict(features, verbose=0)
            emotion_index = prediction.argmax()
            emotion = labels[emotion_index]

            # Draw a rectangle around the face and put the emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame, emotion, (x, y - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2
            )

        # Show the video feed with detections
        cv2.imshow('Emotion Detector', frame)

        # Play the audio for the detected emotion
        if emotion:
            print(f"Detected emotion: {emotion}")
            play_audio(emotion)  # Plays associated audio

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release the video capture object and close all OpenCV windows
    vid.release()
    cv2.destroyAllWindows()
