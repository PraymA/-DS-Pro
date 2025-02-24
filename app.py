from flask import Flask, render_template, request
import pandas as pd
import pickle
import tensorflow as tf

# Пути к моделям
MODEL_PATH = "models/model_NN.keras"
PREPROCESSOR_PATH = "models/Preprocessor_NN.pkl"

# Загрузка модели и препроцессора при старте сервера
model_NN = tf.keras.models.load_model(MODEL_PATH)
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor_NN = pickle.load(f)

# Flask приложение
app = Flask(__name__)

# Определение необходимых параметров
FEATURES = {
    'var2': 'Плотность, кг/м3',
    'var3': 'Модуль упругости, ГПа',
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

# Функция обработки входных данных
def process_input_data(params):
    """ Обрабатывает входные данные из формы, валидирует их и готовит для предсказания. """
    data = {}
    errors = []

    for key, label in FEATURES.items():
        value = params.get(key, "").strip()
        if value:
            try:
                data[key] = float(value)
            except ValueError:
                errors.append(f"{label} - некорректное значение: {value}")
        else:
            errors.append(f"{label} - значение отсутствует")

    return data, errors

@app.route("/", methods=["GET", "POST"])
def predict_page():
    prediction = None
    errors = []
    input_data = {key: "" for key in FEATURES}  # Пустая форма

    if request.method == "POST":
        input_data.update(request.form.to_dict())  # Обновление значений из формы
        processed_data, errors = process_input_data(request.form)

        if not errors:
            # Подготовка данных для модели
            df_input = pd.DataFrame([processed_data])
            df_transformed = preprocessor_NN.transform(df_input)
            prediction = model_NN.predict(df_transformed)[0][0]

    return render_template("index.html", params=input_data, error=errors, result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
