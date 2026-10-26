#!/usr/bin/env python

import requests


url = 'http://localhost:9696/predict'

passenger_id = 'passenger-001'
passenger = {
    'pclass': 3,
    'sex': 'female',
    'age': 22,
    'sibsp': 1,
    'parch': 0,
    'fare': 7.25,
    'embarked': 'S',
}

response = requests.post(url, json=passenger, timeout=10)
response.raise_for_status()
prediction = response.json()
print(prediction)

if prediction['survived']:
    print(f'{passenger_id} is predicted to survive.')
else:
    print(f'{passenger_id} is predicted not to survive.')
