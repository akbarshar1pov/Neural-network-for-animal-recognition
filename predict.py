import os
import cv2
import keras
import numpy as np

# Создается список категорий (CATEGORIES), каждый элемент которого соответствует определенному классу животного.
CATEGORIES = ['chicken', 'cat', 'dog', 'cow', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']


# Определяется функция image, которая считывает изображение из указанного пути, изменяет его размер до 200x200 пикселей
# и возвращает массив numpy.


def image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    new_arr = cv2.resize(img, (200, 200))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 200, 200, 3)
    return new_arr


# Загружается предварительно обученная модель (model), которая будет использоваться для предсказания класса изображения.
# название модели у вас можеть быть другой!
model = keras.models.load_model('model-20.02.2023_01.55.model')

# Указывается путь к изображению для которое надо распознать.
PATH = f"F:\\image.jpeg"

# Выполняется предсказание класса изображения с помощью модели model и функции image для каждого изображения в
# директории.
prediction = model.predict([image(os.path.join(PATH))])

# Результат предикта
print("Predict :", CATEGORIES[prediction.argmax()])
