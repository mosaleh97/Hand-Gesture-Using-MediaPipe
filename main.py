# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time

import mediapipe as mp
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result.gestures[0][0]))


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# Get start time in milliseconds start at zero
start_time = time.time()

with GestureRecognizer.create_from_options(options) as recognizer:
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(0)

    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    while cap.isOpened():
        success, image = cap.read()
        # Convert to numpy.ndarray with floaat32 type
        # image = np.asarray(image, dtype=np.float32)
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame received from OpenCV to a MediaPipe’s Image object
        # image = mp.Image(image, use_gpu=True)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,  # Update the image format if necessary
            data=image.astype(np.uint8)  # Convert image data to np.uint8
        )

        # Process the image and obtain the gesture recognition result and set the timestamp with the current time
        end_time = time.time()
        timestamp_ms = int((end_time - start_time) * 1000)
        recognizer.recognize_async(image, timestamp_ms=timestamp_ms)

        # Convert the MediaPipe image back to OpenCV format and display it
        # cv2.imshow('MediaPipe Gesture Recognition', image)

# Convert the frame received from OpenCV to a MediaPipe’s Image object


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
