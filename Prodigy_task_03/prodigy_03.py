import os
import numpy as np
import cv2
import pickle

# Step 1: Define the directory containing the images
dir = '/Users/anuragtiwari/Downloads/pet_image/pet_name'

# Step 2: Define the categories
categories = ['cats', 'dogs']
data = []

# Step 3: Load and process images
for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    
    if not os.path.exists(path):
        print(f"Directory {path} does not exist")
        continue
    
    print(f"Processing category: {category} ({path})")
    
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            pet_img = cv2.imread(imgpath, 0)  # Read the image in grayscale
            
            if pet_img is not None:
                try:
                    pet_img = cv2.resize(pet_img, (50, 50))
                    image = np.array(pet_img).flatten()
                    data.append([image, label])
                except Exception as e:
                    print(f"Error processing file {imgpath}: {e}")
            else:
                print(f"Could not read image {imgpath}")
        else:
            print(f"Skipping non-image file: {imgpath}")

# Check the length of data
print(f"Total number of images processed: {len(data)}")

# Step 4: Save data to a pickle file
pickle_filename = 'data1.pickle'
try:
    with open(pickle_filename, 'wb') as pick_in:
        pickle.dump(data, pick_in)
    print(f"Data successfully saved to {pickle_filename}")
except Exception as e:
    print(f"Error saving data to {pickle_filename}: {e}")


import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt

# Step 5: Load data from the pickle file
pickle_filename = 'data1.pickle'
try:
    with open(pickle_filename, 'rb') as pick_in:
        data = pickle.load(pick_in)
    print(f"Data successfully loaded from {pickle_filename}")
except Exception as e:
    print(f"Error loading data from {pickle_filename}: {e}")
    data = []

# Check if data is loaded correctly
if len(data) == 0:
    print("No data loaded from pickle file")
else:
    print(f"Loaded {len(data)} samples from pickle file")

# Step 6: Shuffle data
random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.50)

# Step 8: Train the model
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(X_train, y_train)

# Step 9: Save the model to a file
model_filename = 'model.sav'
try:
    with open(model_filename, 'wb') as pick:
        pickle.dump(model, pick)
    print(f"Model successfully saved to {model_filename}")
except Exception as e:
    print(f"Error saving model to {model_filename}: {e}")

# Step 10: Make predictions and calculate accuracy
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)

categories = ['cats', 'dogs']
print('Accuracy:', accuracy)
print('Prediction is:', categories[predictions[0]])

# Step 11: Display the test image and prediction
mypet = np.array(X_test[0]).reshape(50, 50)
plt.imshow(mypet, cmap='gray')
plt.title(f'Prediction: {categories[predictions[0]]}')
plt.show()
