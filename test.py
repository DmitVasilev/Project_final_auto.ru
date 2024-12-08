# Задаем значение для воспроизводимости результатов
seed = 42

# Создаем матрицу признаков
x = df.drop('price', axis=1)

# Формируем вектор правильных ответов.
y = df['price']

# Делим данные на тренировочную (4/5) и тестовую (1/5) выборки
x_train, x_test, y_train, y_test = model_selection.train_test_split(
      x, y, test_size=0.2, random_state=seed, shuffle=True)

# Сериализуем и запишем индексы тестовой выборки в файл
with open('./file_pkl/test_index.pkl', 'wb') as file:
    pickle.dump(x_test.index.to_list(), file)

# Проверяем результат
print('x train: ', x_train.shape)
print('x test: ', x_test.shape)
print('y train: ', y_train.shape)
print('y test: ', y_test.shape)
