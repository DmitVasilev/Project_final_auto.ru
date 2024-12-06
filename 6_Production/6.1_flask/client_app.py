import pandas as pd
import numpy as np
import requests
import json


if __name__ == '__main__':
    
    # Загружаем датасет 
    df = pd.read_csv('./data/clear_auto_data.csv')

    # Формируем матрицу наблюдений
    X = df.drop(['price'], axis=1)

    # Выбираем случайный индекс строки
    random_row = np.random.randint(0, X.shape[0]-1)

    # Формируем строку с признаками
    request_message=json.dumps(X.iloc[[random_row]].to_dict())
    
    # выполняем POST-запрос на сервер по эндпоинту add с параметром json 
    # в котором передаем строку с признаками (по сути - заполненную анкету с сайта auto.ru)
    r = requests.post('http://localhost:5000/predict', json=request_message)
    
    # выводим статус запроса
    print(r.status_code)
    
    # реализуем обработку результата
    if r.status_code == 200:
        
        # если запрос выполнен успешно (код обработки=200),
        # выводим результат на экран
        print(f'Метка данных из тестового набора признаков: {X.iloc[random_row, 0]}')
        print(f"Для полученных данных предсказанная моделью цена автомобиля составляет: "\
            f" {r.json()['prediction']} рублей")
        
    else:
        
        # если запрос завершён с кодом, отличным от 200, выводим содержимое ответа
        print(r.text)