import os
import cv2
import keras
import numpy as np

CATEGORIES = ['chicken', 'cat', 'dog', 'cow', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']


def image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    new_arr = cv2.resize(img, (200, 200))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 200, 200, 3)
    return new_arr


model = keras.models.load_model('model-20.02.2023_01.55.model')

PATH = "F:\\Projects\\Python\\Neural network for animal recognition\\Train\\horse"
true = []
false = []
for i in os.listdir(PATH):
    prediction = model.predict([image(os.path.join(PATH, i))])
    if CATEGORIES[prediction.argmax()] == 'horse':
        true.append(['horse', os.path.join(PATH, i)])
    else:
        false.append([CATEGORIES[prediction.argmax()], os.path.join(PATH, i)])

f = open("Results_of_predicts\\horse_false.txt", 'w')

for i in false:
    f.write(str(i) + "\n")

f.close()

