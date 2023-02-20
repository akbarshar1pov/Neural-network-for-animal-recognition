import numpy as np
import os
import cv2
import pickle
import random
from tqdm import tqdm

# путь к директории с обучающими изображениями
DIRECTORY = r'F:\Projects\Python\Neural network for animal recognition\Train'
# список категорий изображений
CATEGORIES = ['chicken', 'cat', 'dog', 'cow', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
# размер изображений после обработки
IMG_SIZE = 200

data = []  # создание пустого списка для хранения признаков и меток
index = 1  # инициализация переменной-счетчика категорий

# цикл по категориям изображений
for category in CATEGORIES:
    # путь к директории с изображениями текущей категории
    path = os.path.join(DIRECTORY, category)
    # цикл по изображениям текущей категории с отображением прогресса выполнения
    for img in tqdm(os.listdir(path), desc=f'{index}) Classification {category}s...'):
        img_path = os.path.join(path, img)  # путь к текущему изображению
        label = CATEGORIES.index(category)  # метка класса изображения
        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # загрузка изображения
        new_arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))  # изменение размера изображения
        data.append([new_arr, label])  # добавление признаков и метки в список признаков и меток
        pass
    index += 1  # увеличение значения счетчика категорий

# перемешивание списка признаков и меток случайным образом
random.shuffle(data)

X = []  # создание пустого списка для хранения признаков
y = []  # создание пустого списка для хранения меток

# цикл по списку признаков и меток
for features, label in data:
    X.append(features)  # добавление признаков в список признаков
    y.append(label)  # добавление меток в список меток

X = np.array(X)  # преобразование списка признаков в массив NumPy
y = np.array(y)  # преобразование списка меток в массив NumPy

# С помощью метода pickle.dump() сохраняем массивы X и y в два файла X.pkl и y.pkl соответственно:
pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))

print("Classification completed!")
