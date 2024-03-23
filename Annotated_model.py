import os
import pandas as pd

# Define the directory containing your annotation files
annotation_dir = 'C:/Users/ganas/OneDrive/Desktop/yolov5/yolov5-master/Imgs/OBJECT_DETECTION_JOB/labels'

# Create an empty list to store annotation data
data = []

# Iterate through annotation files in the directory
for filename in os.listdir(annotation_dir):
    with open(os.path.join(annotation_dir, filename), 'r') as file:
        for line in file:
            parts = line.strip().split()
            label = parts[0]
            x_min = parts[1]
            y_min = parts[2]
            x_max = parts[3]
            y_max = parts[4]
            data.append([label, x_min, y_min, x_max, y_max])

# Create a DataFrame from the annotation data
df = pd.DataFrame(data, columns=['label', 'x_min', 'y_min', 'x_max', 'y_max'])

# Save the DataFrame as a CSV file
df.to_csv('annotations2.csv', index=False)
