# auto-ml

an automated machine learning pipeline that takes you from raw data to a production-ready model with just one command.

## how it works

`auto-ml` automates the entire machine learning workflow:

- **data cleaning**: cleans your data by handling missing values, duplicates, inconsistent types, etc.
- **feature engineering**: automatically engineers features from numeric, categorical, datetime, and text columns.
- **feature selection**: selects the most important features to improve model performance and reduce noise.
- **model training**: trains multiple models to find the best one for your task.
- **hyperparameter optimization**: uses Optuna to tune each model for optimal results.
- **evaluation**: evaluates the best model on a hold-out test set to see how it performs on unseen data.
- **reporting**: generates a detailed html report with shap analysis, feature importance, and learning curves.
- **customization**: allows for detailed configuration of each step, including model selection, via a YAML file.

whether you're working on a classification or regression problem, the pipeline automatically detects the task and configures itself accordingly.

> **note:** every dataset is unique and may require special treatment. while the default configuration works for most cases, you may need to adjust it to better suit your specific data. see the [configuration](#configuration) section for details.

## why use this?

- **best practices built-in**: the pipeline automatically handles data cleaning, cross-validation, and proper train-test splits.
- **fully configurable**: customize every step of the pipeline through a simple YAML file.
- **explainable results**: understand your model with detailed reports, including SHAP analysis and feature importance.
- **production-ready**: instantly deploy your trained models as a REST API with a single command, either locally or with Docker.
- **save time**: go from a raw dataset to a trained model in minutes
> **note**: training time varies based on dataset size, number of models, and optimization trials. small datasets may take minutes, while larger ones with extensive tuning can take hours.

## getting started

### installation

**1. clone the repository**

**https**
```bash
git clone https://github.com/404brtk/auto-ml.git
cd auto-ml
```

**ssh**
```bash
git clone git@github.com:404brtk/auto-ml.git
cd auto-ml
```

**2. install dependencies**

to get started, you'll need python 3.11+ and `uv`.

install the project and its dependencies:
```bash
# install uv (if you don't have it)
pip install uv

# sync dependencies
uv sync
```

### usage

**1. train a model**

to run the full training pipeline, use the `run` command. provide the path to your dataset and the name of the column you want to predict.

```bash
uv run auto-ml run --dataset /path/to/your/data.csv --target your_target_column
```

the pipeline will handle the rest. once finished, it will save the trained model, performance metrics, and a detailed report to an `outputs` directory.

**2. deploy the model**

after a successful run, deploy the best model as a REST API with the `deploy` command. just point it to the run directory created in the previous step.

```bash
uv run auto-ml deploy /path/to/your/outputs/run_...
```

this will start a web server, and you can begin sending requests to your model to get predictions.

**3. make predictions**

send a post request with one or more samples in the `inputs` list to get predictions.

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "inputs": [
         { "feature1": 10, "feature2": "value1" },
         { "feature1": 20, "feature2": "value2" }
       ]
     }'
```

## using with docker

you can also build and run the application using Docker.

**1. build the docker image**
```bash
docker build -t auto-ml .
```

**2. run commands**

mount your data and output directories to run the pipeline inside the container.

- **train a model:**
  ```bash
  docker run --rm \
    -v ${PWD}/data:/app/data \
    -v ${PWD}/outputs:/app/outputs \
    auto-ml run --dataset /app/data/your_data.csv --target your_target_column
  ```

- **deploy the api:**
  ```bash
  docker run --rm -p 8000:8000 \
    -v ${PWD}/outputs:/app/outputs \
    auto-ml deploy /app/outputs/run_... --host 0.0.0.0 --port 8000
  ```

## what you'll get

after training, you'll find the following in your `outputs/run_*` directory:

**model artifacts:**
- `production_model.joblib`: the final model pipeline, retrained on the full dataset, ready for deployment.
- `eval_model.joblib`: the model trained for evaluation purposes (on the training set only).
- `label_encoder.joblib`: the label encoder (for classification tasks with text labels).
- `name_mapping.json`: a mapping of original to cleaned feature names.
- `results.json`: a summary of the training run, including the best model, parameters, and all metric scores.

**html report (`training_report.html`):**
- **model performance**: a comparison table of all models, plus detailed metrics for the best one.
- **performance plots**:
    - **classification**: confusion matrix, roc curve, and precision-recall curve.
    - **regression**: actual vs. predicted plot and a residuals plot.
- **feature analysis**:
    - **feature importance**: ranks features using built-in and permutation-based methods.
    - **shap explainability**: understand feature contributions with shap summary and dependency plots.
- **learning curves**: visualize how the model's performance changes with more training data.

**rest api (after deployment):**
- `GET /health`: service health check.
- `POST /predict`: get predictions for new data.
- `GET /docs`: interactive api documentation.

## supported models

the pipeline supports a wide range of models, which are specified in the configuration file by their short names.

- **linear models**: `logistic`, `linear`, `ridge`, `lasso`, `elastic_net`, `sgd`
- **probabilistic models**: `naive_bayes`
- **instance-based models**: `knn`, `svm`
- **tree-based models**: `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`, `adaboost`, `xgboost`, `lightgbm`, `catboost`
- **neural network models**: `mlp`

## configuration

for more advanced use cases, you can customize the pipeline by providing a configuration file. see the `configs` directory for examples:

- **`configs/default.yaml`**: a comprehensive configuration with all available options and detailed comments. use this as a reference to understand what can be customized.
- **`configs/quickstart.yaml`**: a minimal configuration to get started quickly with sensible defaults.

to use a custom configuration, simply pass it to the `run` command:
```bash
uv run auto-ml run --dataset /path/to/data.csv --target your_target_column --config /path/to/config.yaml
```

## contributing

we welcome contributions! if you have ideas for new features or have found a bug, please open an issue or submit a pull request.

## license

this project is licensed under the MIT License. see the `LICENSE` file for more details.
