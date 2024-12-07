import pika
import json
import pickle
import numpy as np
import pandas as pd
from func.myfunc import *

# Читаем файл с сериализованным пайплайном
with open('./file_pkl/pipe.pkl', 'rb') as pkl_file:
    loaded_pipe = pickle.load(pkl_file)


# Создаём функцию callback для обработки данных из очереди y_pred
def callback(ch, method, properties, body):
    """Функция для обраюотки сообщения из очереди"""

    # Десериализуем сообщение и извлекаем метку (id) и признаки
    message_body = json.loads(body)
    message_id = message_body['id']
    features = pd.DataFrame(message_body['body'])

    # Делаем предсказание
    pred = loaded_pipe.predict(features)

    # Формируем сообщение с предсказанием
    message_y_pred = {'id': message_id,
                      'body': np.exp(pred[0])-1
                      }

    # Публикуем сообщение в очередь
    channel.basic_publish(exchange='',
                          routing_key='y_pred',
                          body=json.dumps(message_y_pred)
                          )

    print(f'Предсказание {np.exp(pred[0])-1} отправлено в очередь y_pred')


try:
    # Создаём подключение к серверу на локальном хосте:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()

    # Объявляем очередь features
    channel.queue_declare(queue='features')

    # Объявляем очередь y_pred
    channel.queue_declare(queue='y_pred')

    # Извлекаем сообщение из очереди features
    # on_message_callback показывает, какую функцию вызвать при получении
    # сообщения
    channel.basic_consume(queue='features',
                          on_message_callback=callback,
                          auto_ack=True
                          )

    print('...Ожидание сообщений, для выхода нажмите CTRL+C')

    # Запускаем режим ожидания прихода сообщений
    channel.start_consuming()

except Exception as error:
    print('Не удалось подключиться к очереди: {}'.format(error))
