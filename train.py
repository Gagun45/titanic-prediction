import pickle

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


DATA_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
OUTPUT_FILE = 'model.bin'


def main():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.lower()
    features = [
        'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived'
    ]
    df = df[features].copy()
    df['embarked'] = df['embarked'].fillna('Unknown')
    df['pclass'] = df['pclass'].map({1: 'first', 2: 'second', 3: 'third'})
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df = df.drop(columns=['sibsp', 'parch'])

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1
    )
    # Reproduce the notebook's training-only age mean.
    df_train, _ = train_test_split(
        df_full_train, test_size=0.25, random_state=1
    )
    mean_age = df_train['age'].mean()
    df_full_train = df_full_train.copy()
    df_test = df_test.copy()
    df_full_train['age'] = df_full_train['age'].fillna(mean_age)
    df_test['age'] = df_test['age'].fillna(mean_age)

    y_full_train = df_full_train['survived'].to_numpy()
    y_test = df_test['survived'].to_numpy()
    full_train_features = df_full_train.drop(columns='survived')
    test_features = df_test.drop(columns='survived')

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(full_train_features.to_dict(orient='records'))
    X_test = dv.transform(test_features.to_dict(orient='records'))
    feature_names = list(dv.get_feature_names_out())
    dfulltrain = xgb.DMatrix(
        X_full_train, label=y_full_train, feature_names=feature_names
    )
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

    params = {
        'eta': 0.3,
        'max_depth': 8,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }
    model = xgb.train(params, dfulltrain, num_boost_round=17)
    y_pred = model.predict(dtest)
    print(f'Test AUC: {roc_auc_score(y_test, y_pred):.5f}')

    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump(
            {'vectorizer': dv, 'model': model, 'mean_age': mean_age},
            f_out,
        )
    print(f'The model is saved to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
