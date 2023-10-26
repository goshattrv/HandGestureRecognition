import os
import cv2

# Define the directory for storing data
DATA_DIR = './dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes and images per class
num_classes = 36
num_images = 50

# Initialize the camera
cap = cv2.VideoCapture(0)

# Loop through each class
for class_idx in range(num_classes):
    class_dir = os.path.join(DATA_DIR, str(class_idx))

    # Create a directory for the current class if it doesn't exist
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_idx}')

    # Wait for user input to start capturing images
    done = False
    while not done:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "F" ! :)', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('f') or key == ord('F'):
            done = True

    # Capture and save images for the current class
    counter = 0
    while counter < num_images:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        img_filename = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_filename, frame)
        counter += 1

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
