import pickle

import xgboost as xgb
from flask import Flask, jsonify, request


MODEL_FILE = 'model.bin'

with open(MODEL_FILE, 'rb') as f_in:
    artifact = pickle.load(f_in)

dv = artifact['vectorizer']
model = artifact['model']
mean_age = artifact['mean_age']
feature_names = list(dv.get_feature_names_out())

app = Flask('titanic')


@app.route('/predict', methods=['POST'])
def predict():
    passenger = request.get_json()

    features = {
        'pclass': {1: 'first', 2: 'second', 3: 'third'}[passenger['pclass']],
        'sex': passenger['sex'],
        'age': mean_age if passenger.get('age') is None else passenger['age'],
        'fare': passenger['fare'],
        'embarked': passenger.get('embarked') or 'Unknown',
        'family_size': passenger['sibsp'] + passenger['parch'] + 1,
    }

    X = dv.transform([features])
    data = xgb.DMatrix(X, feature_names=feature_names)
    survival_probability = float(model.predict(data)[0])

    return jsonify({
        'survival_probability': survival_probability,
        'survived': survival_probability >= 0.5,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)
