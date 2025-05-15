import torch
import numpy as np
from torch.utils.data import Dataset
from opendataval.util import set_random_state
import os

from typing import Any, Optional
from matplotlib.axes import Axes
from torch.utils.data import Subset, Dataset
from opendataval.dataloader import DataFetcher
from opendataval.dataval import DataEvaluator
from opendataval.metrics import Metrics
from opendataval.model import Model
from opendataval.util import get_name

def evaluate_model(model, x_test, y_test):

    x_test, y_test = process_data_for_sklearn(x_test, y_test)
    
    y_pred = model.predict_proba(x_test)
    predictions = np.argmax(y_pred, axis=1)
    
    accuracy = np.mean(predictions == y_test)
    
    return accuracy

def set_seed(seed=42):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random_state = set_random_state(seed) 


def create_output_dir(path):

    os.makedirs(path, exist_ok=True)

def process_data_for_sklearn(x, y):

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(x, Dataset):
        x = np.array(x)
        
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(y, Dataset):
        y = np.array(y)
        

    if len(y.shape) == 1:
        return x, y
        

    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    elif len(y.shape) > 1 and y.shape[1] == 1:
        y = y.ravel()
        
    return x, y

def train_model_on_subset(model, x, y, train_kwargs, device=None):

    x, y = process_data_for_sklearn(x, y)
    
    if 'sample_weight' in train_kwargs:
        model.fit(x, y, sample_weight=train_kwargs['sample_weight'])
    else:
        model.fit(x, y)
    
    return model


def remove_points_one_by_one(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    model: Optional[Model] = None,
    data: Optional[dict[str, Any]] = None,
    plot: Optional[Axes] = None,
    metric: Metrics = Metrics.ACCURACY,
    train_kwargs: Optional[dict[str, Any]] = None
) -> dict[str, list[float]]:

    if isinstance(fetcher, DataFetcher):
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints
    else:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    data_values = evaluator.data_values
    model = model if model is not None else evaluator.pred_model
    curr_model = model.clone()

    num_points = len(x_train)
    sorted_value_list = np.argsort(data_values)
    
    valuable_list, unvaluable_list = [], []
    train_kwargs = train_kwargs if train_kwargs is not None else {}

    for i in range(num_points):
        if i % 100 == 0:
            print(f"Processing point {i}/{num_points}")
            

        most_valuable_indices = sorted_value_list[i:]
        valuable_model = curr_model.clone()
        valuable_model.fit(
            Subset(x_train, most_valuable_indices),
            Subset(y_train, most_valuable_indices),
            **train_kwargs
        )
        y_hat_valid = valuable_model.predict(x_test).to("cpu")
        valuable_score = metric(y_test, y_hat_valid)
        valuable_list.append(valuable_score)


        least_valuable_indices = sorted_value_list[:(num_points-i)]
        unvaluable_model = curr_model.clone()
        unvaluable_model.fit(
            Subset(x_train, least_valuable_indices),
            Subset(y_train, least_valuable_indices),
            **train_kwargs
        )
        y_hat_valid = unvaluable_model.predict(x_test).to("cpu")
        unvaluable_score = metric(y_test, y_hat_valid)
        unvaluable_list.append(unvaluable_score)

    x_axis = [i/num_points for i in range(num_points)]
    
    if plot is not None:
        plot.plot(x_axis, valuable_list, "o-", label="Remove most valuable first")
        plot.plot(x_axis, unvaluable_list, "x-", label="Remove least valuable first")
        plot.set_xlabel("Fraction Removed")
        plot.set_ylabel(get_name(metric))
        plot.legend()
        plot.set_title(str(evaluator))

    return {
        f"remove_least_influential_first_{get_name(metric)}": valuable_list,
        f"remove_most_influential_first_{get_name(metric)}": unvaluable_list,
        "axis": x_axis,
    }

