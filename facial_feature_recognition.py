# import cv2
# import dlib
# import numpy as np

# # Initialize dlib's face detector (HOG-based) and the facial landmarks predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("C:/Users/ganas/OneDrive/Desktop/yolov5/yolov5-master/Skincare_recomm/final_project/Dash_App_and_Models/shape_predictor_68_face_landmarks.dat")  # Download this file from dlib's website

# # Function to perform face detection and feature extraction
# def extract_facial_features(image_path):
#     # Read the image using OpenCV
#     img = cv2.imread(image_path)
#     print("img",img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = detector(gray)

#     for face in faces:
#         # Get facial landmarks for each detected face
#         landmarks = predictor(gray, face)

#         # Initialize an array to store the facial feature coordinates
#         facial_feature_coords = []

#         # Extract the facial landmarks and store their coordinates
#         for i in range(68):  # Assuming 68 facial landmarks (adjust if using a different predictor)
#             x = landmarks.part(i).x
#             y = landmarks.part(i).y
#             facial_feature_coords.append((x, y))

#         # You can further process the facial_feature_coords or use it directly for analysis

#     return facial_feature_coords  # Return the coordinates of facial landmarks

# # Loop through images in the dataset and perform face detection and feature extraction
# # image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Replace with your image file paths
# image_paths = "C:/Users/ganas/OneDrive/Desktop/yolov5/yolov5-master/runs/detect/yolo_skin_det2/1_0.jpg.jpg"
# for image_path in image_paths:
#     facial_feature_coords = extract_facial_features(image_path)
#     # Perform further processing or analysis with the extracted features

# # Example: Displaying facial landmarks on an image
# sample_image_path = 'C:/Users/ganas/OneDrive/Desktop/yolov5/yolov5-master/Skincare_recomm/final_project/sample_img.jpg'  # Replace with an image path from your dataset
# facial_feature_coords = extract_facial_features(sample_image_path)

# img = cv2.imread(sample_image_path)
# for x, y in facial_feature_coords:
#     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)  # Draw circles at the detected facial landmarks

# cv2.imshow("Facial Landmarks", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import cv2
import face_recognition
import os
import numpy as np

# Define the directory containing your image dataset
image_dir = "C:/Users/ganas/OneDrive/Desktop/yolov5/yolov5-master/runs/detect/yolo_skin_det2"

# Initialize empty lists to store detected faces and facial features
detected_faces = []
facial_features = []

# Loop through the image files in the dataset
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        
        # Face detection using OpenCV's Haar Cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            detected_faces.append(face)

            # Facial feature extraction using face_recognition library
            face_encodings = face_recognition.face_encodings(face)
            if len(face_encodings) > 0:
                facial_features.append(face_encodings[0])

# Convert the detected faces and facial features to NumPy arrays
detected_faces = np.array(detected_faces)
facial_features = np.array(facial_features)

# Now you have the detected faces and facial features for further processing
