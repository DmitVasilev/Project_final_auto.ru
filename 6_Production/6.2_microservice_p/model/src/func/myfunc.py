import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import pickle

# Создадим класс кастомного трансформера для получения координат по
# наименованию города
class sity_to_loc_transform(BaseEstimator, TransformerMixin):
    """Кастомный трансформер для получения географических координат по
    наименованию города
    """

    # Создаем функцию для инициализации объекта класса
    def __init__(self):
        """Инициализация дополнительных параметров нам не требуется """
        pass

    # Создаем функцию обучения. Обучение нам не требуется,
    # поэтому функция возвращает сам объект класса
    def fit(self, X, y=None):
        """Обучение тоже не требуется, возвращаем сам объект"""
        return self

    # Определяем функцию преобразования
    def transform(self, X, y=None):
        """Метод принимает датафрейм содержащий колонку 'sity' и добавляет к
        исходному датафрейму колонки с координатами 'lat' и 'long'

        Args:
            X (pd.DataFrame): Исходный датафрейм
            y (pd.Series, optional): Вектор правильных ответов. Для
            трансформера не требуется.

        Returns:
            pd.DataFrame: Трансформированный датафрейм
        """

        # Загружаем словарь с координатами из файла
        with open('./file_pkl/coordinate.pkl', 'rb') as pkl_file:
            coordinate = pickle.load(pkl_file)

        # Применяем лямбда функцию к данным для получения координат
        lat = X['sity'].map(lambda x: coordinate[x][0])
        long = X['sity'].map(lambda x: coordinate[x][1])
        subset = pd.DataFrame({'lat': lat,
                               'long': long}
                              )

        # Добавляем данные к исходному датасету
        X = pd.concat([X, subset], axis=1)

        # Возвращаем преобразованный датасет
        return X


# Создадим класс кастомного трансформера для кодирования признаков
class encoder_transform(BaseEstimator, TransformerMixin):
    """Кастомный трансформер выполняющий однократное и бинарное кодирование
    признаков
    """

    # Создаем функцию для инициализации объекта класса
    def __init__(self):
        """Инициализация дополнительных параметров нам не требуется """
        pass

    # Создаем функцию обучения. Обучение нам не требуется,
    # поэтому функция возвращает сам объект класса
    def fit(self, X, y=None):
        """Обучение тоже не требуется, возвращаем сам объект"""
        return self

    # Определяем функцию преобразования
    def transform(self, X, y=None):
        """Метод принимает исходный датафрейм и выполняет однократное и
        бинарное кодирование признаков. Исходные столбцы удаляются.

        Args:
            X (pd.DataFrame): Исходный датафрейм
            y (pd.Series, optional): Вектор правильных ответов. Для
            трансформера не требуется.

        Returns:
            pd.DataFrame: Трансформированный датафрейм
        """

        # Считываем имена признаков для однократного кодирования из файла
        with open('./file_pkl/for_dummy_ls.pkl', 'rb') as pkl_file:
            for_dummy_ls = pickle.load(pkl_file)

        # Закодируем признаки из списка для однократного кодирования применив
        # лямбда функцию. Результат кодирования занесем в словарь dict_dummy
        dict_dummy = {}
        for i in for_dummy_ls:
            dict_dummy[i] = X[i].map(lambda x: 0 if x == 'Не указано' else 1)

        # Закодируем и разделим оставшиеся два признака ('steering_wheel',
        # 'customs') на четыре, аналогичной лямбда функцией
        dict_dummy['steering_wheel_l'] = X['steering_wheel'].map(
            lambda x: 1 if x == 'Левый' else 0)
        dict_dummy['steering_wheel_r'] = X['steering_wheel'].map(
            lambda x: 1 if x == 'Правый' else 0)
        dict_dummy['customs_pts'] = X['customs'].map(
            lambda x: 1 if x == 'Растаможен' else 0)
        dict_dummy['customs_no_pts'] = X['customs'].map(
            lambda x: 1 if x == 'Растаможен, нет ПТС' else 0)

        # Удалим из исходного датасета изначальные (мы их уже закодировали)
        # признаки
        X.drop(for_dummy_ls, inplace=True, axis=1)
        X.drop(['steering_wheel', 'customs', 'sity'], inplace=True, axis=1)

        # Добавим закодированные признаки к исходному датасету
        X = pd.concat([X, pd.DataFrame(dict_dummy)], axis=1)

        # Считываем bin_encoder из файла
        with open('./file_pkl/bin_encoder.pkl', 'rb') as pkl_file:
            bin_encoder = pickle.load(pkl_file)

        # Создаем список признаков для бинарного кодирования
        for_binary_ls = ['availability',
                         'transmission',
                         'drive',
                         'condition',
                         'owners',
                         'pts',
                         'eng_type',
                         'body',
                         'color',
                         'brand',
                         'model',
                         'generation'
                         ]

        # Кодируем признаки
        data = bin_encoder.transform(X[for_binary_ls])

        # Удаляем исходные признаки и добавляем результат кодирования
        X = pd.concat([X.drop(for_binary_ls, axis=1), data], axis=1)

        # Возвращаем датасет с закодированными признаками
        return X

# Создадим класс кастомного трансформера для robust преобразования признаков


class robust_scaler(BaseEstimator, TransformerMixin):
    """Кастомный трансформер удаляющий признаки с высокой силой корреляции,
    выстраивающий правильный порядок признаков и применяющий обученный
    трансформер RobustScaler
    """

    # Создаем функцию для инициализации объекта класса
    def __init__(self):
        """Инициализация дополнительных параметров нам не требуется """
        pass

    # Создаем функцию обучения. Обучение нам не требуется,
    # поэтому функция возвращает сам объект класса
    def fit(self, X, y=None):
        """Обучение тоже не требуется, возвращаем сам объект"""
        return self

    # Определяем функцию преобразования
    def transform(self, X, y=None):
        """Метод принимает исходный датафрейм. Удаляет признаки с высокой
        силой корреляции, выстраивает правильный порядок признаков и применяет
        обученный трансформер RobustScaler

        Args:
            X (pd.DataFrame): Исходный датафрейм
            y (pd.Series, optional): Вектор правильных ответов. Для
            трансформера не требуется.

        Returns:
            pd.DataFrame: Трансформированный датафрейм
        """

        # Загружаем список признаков с высокой корреляцией из файла
        with open('./file_pkl/feature_corr.pkl', 'rb') as pkl_file:
            feature_corr = pickle.load(pkl_file)

        # Удаляем признаки с высокой корреляцией из набора данных
        X = X.drop(feature_corr, axis=1)

        # Загружаем список c порядком столбцов из файла
        with open('./file_pkl/col_order.pkl', 'rb') as pkl_file:
            col_order = pickle.load(pkl_file)

        # Восстанавливаем порядок столбцов
        X = X[col_order]

        # Загружаем обученный r_scaler из файла
        with open('./file_pkl/r_scaler.pkl', 'rb') as pkl_file:
            r_scaler = pickle.load(pkl_file)

        # Нормализуем признаки
        X = pd.DataFrame(r_scaler.transform(X=X), columns=X.columns)

        # Возвращаем преобразованный датасет
        return X


# Создадим класс кастомного трансформера для выбора признаков
class best_features(BaseEstimator, TransformerMixin):
    """ Кастомный трансфориер предназначен для выбора определенных
    заранее признаков
    """
    # Создаем функцию для инициализации объекта класса
    def __init__(self):
        """Инициализация дополнительных параметров нам не требуется """
        pass

    # Создаем функцию обучения. Обучение нам не требуется,
    # поэтому функция возвращает сам объект класса
    def fit(self, X, y=None):
        """Обучение тоже не требуется, возвращаем сам объект"""
        return self

    # Определяем функцию преобразования
    def transform(self, X, y=None):
        """Метод принимает исходный датафрейм и возвращает срез отобранных
        с помощью дерева решений признаков

        Args:
            X (pd.DataFrame): Исходный датафрейм
            y (pd.Series, optional): Вектор правильных ответов. Для
            трансформера не требуется.

        Returns:
            pd.DataFrame: Трансформированный датафрейм
        """

        # Загружаем список с отобранными признаками из файла
        with open('./file_pkl/best_dt_importance.pkl', 'rb') as pkl_file:
            best_dt_importance = pickle.load(pkl_file)

        # Возвращаем преобразованный датасет
        return X[best_dt_importance]