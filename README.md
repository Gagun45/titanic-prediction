# Titanic survival prediction

## Problem description

This project predicts whether a passenger survived the Titanic disaster using
passenger class, sex, age, fare, embarkation port, and family size. It is a binary
classification task: `survived = 1` means the passenger survived, and `0` means
they did not.

The project uses the public, 891-passenger Titanic dataset hosted by
[Data Science Dojo](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv).
The training script and notebook download the CSV automatically. Family size is
calculated from the numbers of siblings/spouses and parents/children aboard;
missing ages are filled using the training split's mean age.


## Guide for peer reviewers

Recommended review order:

1. Open [notebook.ipynb](notebook.ipynb) to inspect data preparation, basic EDA,
   train/validation/test splitting, and model selection. The notebook compares
   decision trees, random forests, and XGBoost, with hyperparameter tuning and
   validation ROC AUC results. Saved outputs allow an initial review without
   running the notebook.
2. Open [train.py](train.py) to check the final XGBoost configuration,
   preprocessing, held-out test evaluation, and artifact saving. Running it
   prints the test ROC AUC and creates `model.bin`, containing the model,
   vectorizer, and age-imputation mean. This generated artifact is ignored by
   Git, so train before building the Docker image.
3. Open [predict.py](predict.py) to inspect the `POST /predict` endpoint and how
   request data is transformed into the model's features.
4. Open [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), and
   [Dockerfile](Dockerfile) to review dependencies and containerization.
5. Follow the commands below to train, build, start the service, and send a
   prediction request. A successful request returns JSON containing
   `survival_probability` (between 0 and 1) and `survived` (a boolean).

For local verification, install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Docker, and start the Docker daemon. The project requires Python 3.14 or
newer; uv can provision Python if needed. Initial setup requires internet access
to download dependencies, the dataset, and container images. `uv sync` creates
the project's `.venv`; `uv run` uses that environment without manual activation.

The scoring rubric is in [docs/README.md](docs/README.md#evaluation-criteria).
Cloud deployment is not included in this submission.

## Run with Docker

Install the locked dependencies and train the model first (training downloads the
public Titanic dataset):

```bash
uv sync --frozen
uv run python train.py
```

Build and run the prediction service, served by Gunicorn on port 9696:

```bash
docker build -t titanic-predict .
docker run --rm -p 9696:9696 titanic-predict
```

Keep the container running. In another terminal, open the project directory and
run the example prediction client:

```bash
uv run python predict-test.py
```

The client sends a passenger's attributes to `http://localhost:9696/predict`,
checks for HTTP errors, and prints the prediction.

Alternatively, send an example request with curl:

```bash
curl --fail-with-body http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{"pclass":3,"sex":"female","age":22,"sibsp":1,"parch":0,"fare":7.25,"embarked":"S"}'
```

The response includes `survival_probability` and a boolean `survived` prediction.
The image includes the trained `model.bin`; retrain and rebuild to update it.
