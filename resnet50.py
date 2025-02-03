import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import cv2
import os

base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_and_preprocess_image(img_path)
        images.append(img)
        label = 1 if 'good' in filename.lower() else 0 
        labels.append(label)
    return np.array(images), np.array(labels)

X, y = load_images_from_folder('image_base')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

features_train = base_model.predict(X_train)
features_test = base_model.predict(X_test)

svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(features_train, y_train)

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def predict_image(image_path):
    img = load_and_preprocess_image(image_path)
    features = base_model.predict(np.expand_dims(img, axis=0))
    svm_prob = svm_classifier.predict_proba(features)[0]
    
  
    original_img = cv2.imread(image_path)
    edges = edge_detection(original_img)
    
  
    
    if svm_prob[1] > 0.5: 
        return "GOOD"
    else:
        return "NOPE"

user_image_path = 'image_user_resize/155347 (1).jpg' 
result = predict_image(user_image_path)
print(result)