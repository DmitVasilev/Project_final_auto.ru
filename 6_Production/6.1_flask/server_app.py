from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json

with open ('./file_pkl/pipe.pkl','rb') as pkl_file:
    loaded_pipe = pickle.load(pkl_file) 

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Извлекаем json из запроса и десериализуем его для получения признаков 
    features = pd.DataFrame(json.loads(request.json))

    # Делаем предсказание
    pred = loaded_pipe.predict(features)
   
    return  jsonify({'prediction': np.exp(pred[0])-1})

if __name__ == '__main__':
    
    app.run('localhost', 5000)