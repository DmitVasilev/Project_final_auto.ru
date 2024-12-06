import pika
import numpy as np
import json
import pandas as pd
import time

# Загружаем датасет 
df = pd.read_csv('./data/clear_auto_data.csv')

# Формируем вектор правильных ответов
y = df['price']

# Формируем матрицу наблюдений
X = df.drop(['price'], axis=1)

# Создаем бесконечный цикл для отправки сообщений в очередь 
while True:
    try:
        # Выбираем случайный индекс строки
        random_row = np.random.randint(0, X.shape[0]-1)

        # Подключение к серверу на локальном хосте:
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
        channel = connection.channel()

        # Создаём очередь y_true
        channel.queue_declare(queue='y_true')
        # Создаём очередь features
        channel.queue_declare(queue='features')
        
        # Создаем метку сообщения из старой колонки с индексами
        message_id = int(X.iloc[random_row, 0])

        # Публикуем сообщение в очередь y_true        
        message_y_true = {'id':message_id,
                          'body':float(y.iloc[random_row])
                          }

        channel.basic_publish(exchange='',
                              routing_key='y_true',
                              body=json.dumps(message_y_true)
                              )
        
        print('Сообщение с правильным ответом отправлено в очередь')

        # Публикуем сообщение в очередь features
        message_features = {'id':message_id,
                            'body':X.iloc[[random_row]].to_dict()
                            }
        
        channel.basic_publish(exchange='',
                              routing_key='features',
                              body=json.dumps(message_features)
                              )
        
        print('Сообщение с вектором признаков отправлено в очередь')

        # Закрываем подключение
        connection.close()
        
        # Делаем задержку
        time.sleep(10)
        
    except Exception as error:
        print('Не удалось подключиться к очереди: {}'.format(error))