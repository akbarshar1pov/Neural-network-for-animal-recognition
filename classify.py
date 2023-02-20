import numpy as np
import os
import cv2
import pickle
import random
from tqdm import tqdm

DIRECTORY = r'F:\Projects\Python\Neural network for animal recognition\Train'
CATEGORIES = ['chicken', 'cat', 'dog', 'cow', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
IMG_SIZE = 200

data = []
index = 1
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in tqdm(os.listdir(path), desc=f'{index}) Classification {category}s...'):
        img_path = os.path.join(path, img)
        label = CATEGORIES.index(category)
        arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        new_arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
        data.append([new_arr, label])
        pass
    index += 1

random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))

print("Classification completed!")
