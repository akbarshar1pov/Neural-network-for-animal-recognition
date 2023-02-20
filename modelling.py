import datetime
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

# Генерируем уникальное имя для модели
NAME = f"animals-prediction-{datetime.datetime.now().date().strftime('%y.%m.%d')}"

# Создаем объект TensorBoard Callback для записи метрик и логов обучения во время обучения модели в Keras
# и указываем дирректорию 'logs/' для записи логов.
tensorboard = TensorBoard(log_dir='logs/')

# Загружаем данные из файлов X.pkl и y.pkl
X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

# Нормализуем данные и изменяем размерность массива X
X = X / 255
X = X.reshape(-1, 200, 200, 1)

# Создаем последовательную модель
model = Sequential()

# Добавляем сверточные и пулинговые слои
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Слой закоментрован для проведения сравнительного анализа обучаемой модели
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))

# Преобразуем данные перед подачей на полносвязные слои
model.add(Flatten())

# Добавляем полносвязные слои
model.add(Dense(128, input_shape=X.shape[1:], activation='relu'))
# Слой закоментрован для проведения сравнительного анализа обучаемой модели
# model.add(Dense(128, activation='relu'))

# Добавляем выходной слой с функцией активации softmax
model.add(Dense(9, activation='softmax'))

# Компилируем модель
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель
model.fit(X, y, epochs=5, validation_split=0.1, batch_size=32, callbacks=[tensorboard])

# Сохраняем обученную модель в файл
model.save(NAME)
print(f"The model was saved under the name '{NAME}'.model")
