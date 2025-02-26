from flask import Flask, render_template, request
import pandas as pd
import pickle
import tensorflow as tf

MODEL_PATH = "models/model_NN_2.keras"
PREPROCESSOR_PATH = "models/Preprocessor_NN_scaler.pkl"

# Загрузка модели и препроцессора при старте сервера
model_NN = tf.keras.models.load_model(MODEL_PATH)
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor_NN = pickle.load(f)

app = Flask(__name__)

def get_data_from_form(features, params):
    param_names = features.keys()
    data = dict.fromkeys(param_names, None)
    error = ''
    # преобразование из строк в числа
    for param_name, param_value in params.items():
        if param_value.strip(' \t') != '':
            try:
                data[param_name] = float(param_value)
            except ValueError:
                error += f'{features[param_name]} - некорректное значение "{param_value}"\n'
    return data, error

@app.route('/', methods=['post', 'get'])
def model_NN_page():
    features = {
        'var2': 'Плотность, кг/м3',
        'var3': 'модуль упругости, ГПа',
        'var4': 'Количество отвердителя, м.%',
        'var5': 'Содержание эпоксидных групп, %',
        'var6': 'Температура вспышки, °С',
        'var7': 'Поверхностная плотность, г/м2',
        'var8': 'Модуль упругости при растяжении, ГПа',
        'var9': 'Прочность при растяжении, МПа',
        'var10': 'Потребление смолы, г/м2',
        'var11': 'Угол нашивки',
        'var12': 'Шаг нашивки',
        'var13': 'Плотность нашивки'
    }
    
    params = dict(zip(features.keys(), ['', '', '', '', '', '', '', '', '', '', '', '']))
    error = ''
    x = pd.DataFrame()
    var1 = ''
    if request.method == 'POST':
        params = request.form.to_dict()
        data, error = get_data_from_form(features, params)
        if error == '':
            try:
                x = pd.DataFrame([data], columns=features.keys())
                x3 = preprocessor_NN.transform(x)
                y3 = model_NN.predict(x3)
                var1 = y3[0] 
            except Exception as e:
                error = f"Ошибка при обработке данных: {str(e)}"
    return render_template('index.html', params=params, error=error, inputs=x.to_html(), var1=var1, features=features)

if __name__ == "__main__":
    app.run(debug=True)

