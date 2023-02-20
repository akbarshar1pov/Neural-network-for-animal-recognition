import os
import cv2
import keras
import numpy as np

# Создается список категорий (CATEGORIES), каждый элемент которого соответствует определенному классу животного.

CATEGORIES = ['chicken', 'cat', 'dog', 'cow', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
ANIMAL_NAME = 'horse'  # Название может быть другой. Пример приведен для тестирования зананий модели!

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

# Указывается путь к папке, в которой находятся изображения для классификации (PATH).
PATH = f"F:\\Projects\\Python\\Neural network for animal recognition\\Train\\{ANIMAL_NAME}"
# Создаются два списка: true - для правильно классифицированных изображений и false - для неправильно
# классифицированных.
true = []
false = []

# Происходит итерация по всем изображениям в указанной директории с помощью функции os.listdir.
for i in os.listdir(PATH):
    # Выполняется предсказание класса изображения с помощью модели model и функции image для каждого изображения в
    # директории.
    prediction = model.predict([image(os.path.join(PATH, i))])

    # Если предсказанный класс соответствует "horse", то имя файла добавляется в список true
    if CATEGORIES[prediction.argmax()] == ANIMAL_NAME:
        true.append([ANIMAL_NAME, os.path.join(PATH, i)])
    # иначе добавляется в список false.
    else:
        false.append([CATEGORIES[prediction.argmax()], os.path.join(PATH, i)])

# Создается файл для записи результатов предсказания для неправильно классифицированных изображений (horse_false.txt)
# в директории Results_of_predicts.
f = open(f"Results_of_predicts\\{ANIMAL_NAME}_false.txt", 'w')

# Итерируется по списку false и каждый элемент списка записывается в файл.
for i in false:
    f.write(str(i) + "\n")

f.close()

