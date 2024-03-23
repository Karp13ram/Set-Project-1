import os
import cv2
# Specify the directory where subfolders (each representing a person) with face images are stored
data_dir = "C:/Users/ganas/OneDrive/Desktop/yolov5/yolov5-master/Extracted_Faces"

# Specify the output directory for preprocessed images
output_dir = "C:/Users/ganas/OneDrive/Desktop/yolov5/yolov5-master/extracted_preprocessedimages"
os.makedirs(output_dir, exist_ok=True)

# Define the target size for resizing
target_size = (224, 224)  # You can adjust this size as needed

# Function to preprocess and augment images
def preprocess_image(image):
    # Resize the image to the target size
    image = cv2.resize(image, target_size)

    # Normalize the image to have values in the range [0, 1]
    image = image / 255.0

    return image

# Recursively loop through subfolders (each representing a person)
for person_folder in os.listdir(data_dir):
    person_folder_path = os.path.join(data_dir, person_folder)
    if os.path.isdir(person_folder_path):
        # Create a directory for the current person in the output directory
        person_output_dir = os.path.join(output_dir, person_folder)
        os.makedirs(person_output_dir, exist_ok=True)

        # Loop through the face images in the current person's folder
        for filename in os.listdir(person_folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Read the image
                image_path = os.path.join(person_folder_path, filename)
                image = cv2.imread(image_path)

                # Perform preprocessing
                preprocessed_image = preprocess_image(image)

                # Save the preprocessed image to the output directory for the current person
                output_path = os.path.join(person_output_dir, filename)
                cv2.imwrite(output_path, preprocessed_image)

print("Preprocessing complete. Preprocessed images are saved in", output_dir)

# import os
# import cv2
# input_directory = 'C:/Users/ganas/OneDrive/Desktop/yolov5/yolov5-master/Extracted_Faces'
# output_directory = 'C:/Users/ganas/OneDrive/Desktop/yolov5/yolov5-master/Preprocessed_Extracted_Faces'
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)
# for root, _, files in os.walk(input_directory):
#     for filename in files:
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Modify the file extensions as needed
#             input_path = os.path.join(root, filename)
#             output_path = os.path.join(output_directory, filename)

#             # Load the image
#             image = cv2.imread(input_path)

#             if image is not None:
#                 # Perform image preprocessing here
#                 # Example: Resize the image to a specific size
#                 # image = cv2.resize(image, (new_width, new_height))

#                 # Save the preprocessed image to the output directory
#                 cv2.imwrite(output_path, image)

#                 print(f"Processed and saved: {output_path}")
#             else:
#                 print(f"Failed to read: {input_path}")

