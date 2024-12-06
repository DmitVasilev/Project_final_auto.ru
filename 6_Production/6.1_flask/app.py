import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


# Create flask app
flask_app = Flask(__name__)

with open ('./file_pkl/pipe.pkl','rb') as pkl_file:
    loaded_pipe = pickle.load(pkl_file) 

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    
    # Загружаем датасет 
    df = pd.read_csv('./data/clear_auto_data.csv')

    # Формируем матрицу наблюдений
    data = df.drop(['price'], axis=1)

    # Выбираем случайный индекс строки
    random_row = np.random.randint(0, data.shape[0]-1)
    
    # Запоминаем метку наблюдения
    id = data.iloc[random_row, 0]
    
    # Делаем предсказание
    pred = round(np.exp(loaded_pipe.predict(data.iloc[[random_row]])[0])-1, 2)
    
    return render_template("index.html",
                           prediction_text = f"Для полученных данных с меткой {id}"\
                           f" рекомендуемая цена автомобиля {pred} рублей"
                           )

if __name__ == "__main__":
    flask_app.run(debug=True)