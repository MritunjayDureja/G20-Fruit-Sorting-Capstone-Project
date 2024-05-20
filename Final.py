import cv2
import os
import pathlib
import cv2
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import serial
import time
from random import randint

def detect_object():
    with open('coco.names', 'rt') as f:
        class_name = f.read().rstrip('\n').split('\n')

    class_color = []
    for i in range(len(class_name)):
        class_color.append((randint(0, 255), randint(0, 255), randint(0, 255)))

    modelPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightPath, modelPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Open the webcam
    cap = cv2.VideoCapture(1)  # 0 corresponds to the default camera, change it if necessary

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=class_color[classId - 1], thickness=2)
                cv2.putText(img, class_name[classId - 1].upper(), (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, class_color[classId - 1], 2)
                if class_name[classId - 1].lower() in ["banana", "pizza", "donut", "sandwich", "bowl"]:
                    # Release the webcam and close the window
                    cap.release()
                    cv2.destroyAllWindows()
                    return class_name[classId - 1].lower()
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    return None

def capture_image():
    save_dir = r"D:\Capstone"
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Accessing the webcam
    cap = cv2.VideoCapture(1)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the captured image
    cv2.imshow('Captured Image', frame)
    cv2.waitKey(2000)  # Display the image for 2 seconds

    # Save the captured image to the specified directory
    image_path = os.path.join(save_dir, "captured_image.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Image saved to: {image_path}")

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()  # Close the window after saving the image
    return image_path

# Assuming 'model.h5' is the saved model file
model_predict = tf.keras.models.load_model('model.h5')
model_predict.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# Establish serial connection (update port and baudrate as per your setup)
ser = serial.Serial('COM3', 9600)  # Update 'COM3' with the correct port
time.sleep(2)  # Wait for Arduino to initialize

while True:
    # Detect object
    detected_item = detect_object()
    if detected_item:
        # Capture an image
        image_path = capture_image()

        # Load the image from the specified path
        img = image.load_img(image_path, color_mode="rgb", target_size=(150, 150), interpolation="nearest")
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255

        images = np.vstack([img])
        classes = model_predict.predict(images, batch_size=10)

        max_prob = np.amax(classes[0])
        predicted_class = np.argmax(classes[0])

        class_labels = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']

        predict_result = class_labels[predicted_class]
        print(f"Predict: {predict_result} ({round(float(max_prob) * 100, 2)}%)")

        # Display the result
        plt.figure(figsize=(6, 6))
        plt.imshow(image.load_img(image_path, color_mode="rgb", target_size=(150, 150), interpolation="nearest"))
        title = f"Predict: {predict_result} ({round(float(max_prob) * 100, 2)}%)"
        plt.title(title, color='black')
        plt.axis('off')
        plt.show(block=False)  # Show the plot without blocking

        # Wait for a moment before closing the window
        # plt.pause(0.5)  # Adjust the delay time as needed
        plt.savefig("plot.png", format='png', bbox_inches='tight')
        plt.close()

        # Take input from the variable predict_result to start servo rotation
        user_input = predict_result

        # Send a signal to start the Arduino code if the input matches
        if user_input.lower() == "rotten banana":
            ser.write(b'start')
        elif user_input.lower() == "fresh banana":
            ser.write(b'start2')
        else:
            print("Invalid input. Please try again.")

        # Keep the Python script running until interrupted
        try:
            while True:
                # Read any incoming data from Arduino
                response = ser.readline().strip().decode('utf-8', errors='ignore')
                print("Arduino:", response)
                if response == "done":
                    break  # Break the loop when the Arduino signals completion
                time.sleep(0.1)  # Adjust the delay as needed
        except KeyboardInterrupt:
            # Close the serial connection when the script is interrupted
            ser.close()
            print("Serial connection closed.")

    # Remove this block as we no longer have the 40-second loop
    time.sleep(40)

# # This block should be unindented to be outside the while loop
# except KeyboardInterrupt:
#     # Close the serial connection when the script is interrupted
#     ser.close()
#     print("Serial connection closed.")
