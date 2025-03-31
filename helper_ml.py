import os
import pickle
import torch
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error



def train_xgb_models(X_train, y_train, X_test, y_test, use_gpu=True, name="xgb_model", objective='regression', colsample_bytree=0.7, 
                 learning_rate=0.1, max_depth=120, n_estimators=1000, num_leaves=100, save_model=True):
    """
    Trains a set of models on the given training data and saves the trained models to disk.

    Args:
        X_train (numpy.ndarray): The features for the training data.
        y_train (numpy.ndarray): The target variable for the training data.
        X_test (numpy.ndarray): The features for the test data.
        y_test (numpy.ndarray): The target variable for the test data.
        use_gpu (bool, optional): Whether to use GPU for training. Defaults to True.
        name (str, optional): The base name to use when saving the models. Defaults to "model".
        objective (str, optional): The objective function to use for the models. Defaults to 'regression'.
        colsample_bytree (float, optional): The fraction of columns to be randomly sampled for each tree. Defaults to 0.5.
        learning_rate (float, optional): The learning rate for the models. Defaults to 0.1.
        max_depth (int, optional): The maximum depth of the trees for the models. Defaults to 120.
        n_estimators (int, optional): The number of trees to fit for the models. Defaults to 1000.
        num_leaves (int, optional): The maximum number of leaves for the models. Defaults to 120.
    """
    # Initialize your regressors
    if use_gpu:
        models = [lgb.LGBMRegressor(objective=objective, colsample_bytree=colsample_bytree, learning_rate=learning_rate,
                                    max_depth=max_depth, n_estimators=n_estimators, device='gpu', num_leaves=num_leaves) for _ in range(y_train.shape[1])]
    else:
        models = [xgb.XGBRegressor(objective=objective, colsample_bytree=colsample_bytree, learning_rate=learning_rate,
                                   max_depth=max_depth, alpha=5, n_estimators=n_estimators, num_leaves=num_leaves) for _ in range(y_train.shape[1])]

    # Fit the models to the training data and make predictions
    y_pred = np.zeros_like(y_test)
    for i, model in enumerate(models):
        model.fit(X_train, y_train[:, i])
        y_pred[:, i] = model.predict(X_test)

        # Save the trained model
        if save_model:
            model.booster_.save_model(f'{name}_{i}.json')

    # Compute the RMSE of the predictions
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")

    return y_pred, models


def load_models(num_models, name="xgb_model", use_gpu=True):
    """
    Loads a set of XGBoost models from disk.

    Args:
        num_models (int): The number of models to load.
        name (str, optional): The base name of the models to load. Defaults to "xgb_model".

    Returns:
        list: A list of the loaded XGBoost models.
    """
    # Initialize an empty list to hold the models
    models = []

    # Load the models
    for i in range(num_models):
        if not use_gpu:
            model = xgb.XGBRegressor()
            model.load_model(f"{name}_{i}.json")
        else:
            model = lgb.Booster(model_file=f"{name}_{i}.json")
        models.append(model)

    return models


def generate_predictions_and_train_second_level_model(X_train, y_train, X_test, y_test, num_models, model, x_categ, X_train_tensor, 
                                                      X_test_tensor, E_return, E_return2, scaling_type, name="xgb_model", second_name="model_level2",
                                                      train_second_level_model=True, use_gpu=True, models=None):
    """
    Generates predictions from the first-level models using the train data, trains a second-level model (if it does not exist), 
    to minimize the loss, saving it to a pickle file for later use and generates the final predictions.

    Args:
        X_train (numpy.ndarray): The features for the training data.
        y_train (numpy.ndarray): The target variable for the training data.
        X_test (numpy.ndarray): The features for the test data.
        y_test (numpy.ndarray): The target variable for the test data.
        num_models (int): The number of first-level models.
        model (torch.nn.Module): The Deep Learning model.
        x_categ (torch.Tensor): The categorical features.
        X_train_tensor (torch.Tensor): The features for the training data (as a tensor).
        X_test_tensor (torch.Tensor): The features for the test data (as a tensor).
        E_return (float): The mean of the target variable.
        E_return2 (float): The variance of the target variable.
        scaling_type (str): The type of scaling used.
        name (str, optional): The base name of the first-level models. Defaults to "xgb_model".
        train_second_level_model (bool, optional): Whether to train the second-level model. Defaults to True.
        use_gpu (bool, optional): Whether to use GPU for the Deep Learning model. Defaults to True.

    Returns:
        numpy.ndarray: The final predictions.
    """
    # Load the first-level models
    if models is None:
        models = load_models(num_models, name)

    # Generate predictions from the first-level models on the training data
    # Most time consuming part
    y_pred_train_lgbm = np.column_stack([model.predict(X_train) for model in models])

    # Generate predictions from the Deep Learning model on the training data
    model.eval()
    with torch.no_grad():
        if use_gpu:
            y_pred_train_dl = model(x_categ[:X_train.shape[0]].cuda(), X_train_tensor.cuda()).cpu().numpy()
        else:
            y_pred_train_dl = model(x_categ[:X_train.shape[0]], X_train_tensor).cpu().numpy()

    # Stack the predictions together to form a new feature set for the second-level model using the training data
    X_train_level2 = np.column_stack([y_pred_train_lgbm, y_pred_train_dl])

    # Convert the labels to a numpy array
    y_train_np = y_train.cpu().numpy()



    # Check if the second-level model exists
    if train_second_level_model or not os.path.exists(f"{second_name}.pkl"):
        # Train the second-level model
        dtrain = xgb.DMatrix(X_train_level2, label=y_train_np)
        params = {'objective': 'reg:squarederror', 'eval_metric': 'mae', 'eta': 0.1, 'max_depth': 6}
        model_level2 = xgb.train(params, dtrain, num_boost_round=1000)

        # Save the second-level model
        with open(f"{second_name}.pkl", 'wb') as f:
            pickle.dump(model_level2, f)
    else:
        # Load the second-level model
        with open(f"{second_name}.pkl", 'rb') as f:
            model_level2 = pickle.load(f)



    # Generate predictions from the first-level models on the test data
    y_pred_test_lgbm = np.column_stack([model.predict(X_test) for model in models])

    # Generate predictions from the Deep Learning model on the test data
    model.eval()
    with torch.no_grad():
        if use_gpu:
            y_pred_test_dl = model(x_categ[:X_test.shape[0]].cuda(), X_test_tensor.cuda()).cpu().numpy()
        else:
            y_pred_test_dl = model(x_categ[:X_test.shape[0]], X_test_tensor).cpu().numpy()

    # Stack the predictions together to form a new feature set for the second-level model
    X_test_level2 = np.column_stack([y_pred_test_lgbm, y_pred_test_dl])



    # Generate the final prediction from the second-level model
    dtest = xgb.DMatrix(X_test_level2)
    y_pred_final = model_level2.predict(dtest)

    return y_pred_final