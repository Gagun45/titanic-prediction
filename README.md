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

Reviewers can run and test the service locally using the Docker instructions
below. Deploying it to AWS is optional; the cloud deployment section documents
how the hosted service was deployed for anyone who wants to reproduce it.

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
5. Follow [Run with Docker](#run-with-docker) to train, build, start the service locally, and send a
   prediction request. A successful request returns JSON containing
   `survival_probability` (between 0 and 1) and `survived` (a boolean).

For local verification, install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Docker, and start the Docker daemon. The project requires Python 3.14 or
newer; uv can provision Python if needed. Initial setup requires internet access
to download dependencies, the dataset, and container images. `uv sync` creates
the project's `.venv`; `uv run` uses that environment without manual activation.

The scoring rubric is in [docs/README.md](docs/README.md#evaluation-criteria).
The service has also been deployed and tested on AWS Elastic Beanstalk; see the
deployment instructions below.

## Run with Docker

This is the local deployment and verification path for reviewers. It requires
Docker and uv, with no AWS account or cloud environment needed.

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

### Alternative: run locally with the EB CLI

If you have already initialized the project with `eb init` (see the optional AWS
section below), you can use the EB CLI to build and run the same Docker service
locally on Linux or macOS:

```bash
uv sync --frozen
uv run python train.py
uv run eb local run --port 9696
```

Keep Docker running and stop any other container using port 9696 first. In a
second terminal, run `uv run python predict-test.py`, or send the curl request
above to `http://localhost:9696/predict`. You can also send that request from
Postman using **POST** and **Body → raw → JSON**. Press Ctrl+C in the terminal
running `eb local run` to stop the service.

This command runs the container on your computer and does not create a cloud
environment. AWS requires an initialized EB project for
[`eb local`](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb3-local.html);
reviewers can use the direct Docker commands above to skip AWS setup entirely.

## Optional: deploy to AWS Elastic Beanstalk

This section is optional and is not required for local testing or peer review.
Run these commands locally from the project directory to deploy to AWS.
You need an AWS account
with permission to create Elastic Beanstalk resources and the
[AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
installed. The EB CLI was added as a development dependency with
`uv add awsebcli --dev`; it is already included in this project's lockfile, so
`uv sync --frozen` installs it. Use `uv run eb` to run it without activating the
virtual environment.

Configure an AWS credentials profile and verify access:

```bash
aws configure --profile eb-cli
aws sts get-caller-identity --profile eb-cli
```

Enter your IAM access key ID, secret access key, default region `eu-north-1`, and
output format `json` when prompted. If your account uses IAM Identity Center
(SSO), use `aws configure sso --profile eb-cli` and
`aws sso login --profile eb-cli` instead. Keep credentials outside the repository.

Generate the model locally, then initialize the Elastic Beanstalk application
with the Docker platform in Stockholm (`eu-north-1`):

```bash
uv sync --frozen
uv run python train.py
uv run eb init -p docker -r eu-north-1 titanic-serving --profile eb-cli
```

The Dockerfile copies `model.bin` into the image. The model remains ignored by
Git, while [.ebignore](.ebignore) includes it in the deployment bundle along with
the Docker build inputs. This also means deployment uses the current local
files, including uncommitted changes. See the
[AWS EB CLI configuration guide](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-configuration.html).

For the first deployment, create an environment (choose a different environment
name if needed):

```bash
uv run eb create titanic-serving
uv run eb status titanic-serving
```

`eb create` provisions the AWS resources and deploys the application. For later
updates, use `eb deploy` against the existing environment; regenerate the model
first if you changed training:

```bash
uv run python train.py
uv run eb deploy titanic-serving
```

### Test the deployed service

Copy the environment's CNAME from `uv run eb status titanic-serving` and replace
`YOUR_ENVIRONMENT_CNAME` below. Send requests to `/predict` using the public HTTP
endpoint; port 9696 is used inside the container.

```bash
curl --fail-with-body http://YOUR_ENVIRONMENT_CNAME/predict \
  -H 'Content-Type: application/json' \
  -d '{"pclass":3,"sex":"female","age":22,"sibsp":1,"parch":0,"fare":7.25,"embarked":"S"}'
```

In Postman, select **POST**, enter the same URL, and choose **Body → raw → JSON**.
Use the passenger JSON from the curl example. A successful response has status
`200 OK` and contains `survival_probability` and `survived`. The application only
defines `POST /predict`, so opening `/` in a browser returns 404.

Postman screenshots showing the deployed endpoint, request, and response will be
added here.

If deployment fails, retrieve the full logs with
`uv run eb logs titanic-serving --all` and inspect `eb-engine.log` for the failed
Docker build step. When you no longer need the environment, remove its running
resources with `uv run eb terminate titanic-serving` to avoid ongoing charges.
