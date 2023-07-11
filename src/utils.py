import json
import os.path

import torch

import transforms
from models import cnn, resnet, lcnn


def get_transform(feature_name, **feature_kwargs):
    """
    Given a transform name, retrieves it from defined transforms and instantiates it with given keyword arguments
    :param feature_name: name of the transform, e.g. 'Spectrogram', 'MelSpectrogram', 'MFCC' ...
    :param feature_kwargs: keyword arguments to feed the transform module, e.g. 'n_fft', 'n_filters' ...
    :return: instantiated transform function as a nn.Module object
    """
    try:
        transform = getattr(transforms, feature_name)
    except AttributeError as err:
        print(f"Feature '{feature_name}' could not be found.")
        raise err.with_traceback(err.__traceback__)

    return transform(**feature_kwargs)


def get_model(model_name, *model_args, **model_kwargs):
    """
    Given a model name, retrieves it from defined models and instantiates it with given arguments
    :param model_name: name of the model to use, e.g. 'CNN', 'ResNet' ...
    :param model_args: non-keyword arguments to feed the model constructor
    :param model_kwargs: keyword argument to feed the model constructor
    :return: instantiated and initialized torch model as nn.Module object
    """
    found = False
    for module in [cnn, resnet, lcnn]:
        try:
            model = getattr(module, model_name)
            found = True
            break
        except AttributeError:
            continue
    if not found:
        raise AttributeError(f"Model '{model_name}' could not be found.")

    return model(*model_args, **model_kwargs)


def get_optimizer(optim_name, model, lr, **optimizer_kwargs):
    """
    Given an optimizer name, retrieves it from torch optimizers and instantiates it with given keyword arguments
    :param optim_name: name of the optimizer to use, e.g. 'Adam', 'SGD', ...
    :param model: model to optimize parameters of
    :param lr: learning rate to use for the optimizer's weight updates
    :param optimizer_kwargs: other keyword arguments to feed the optimizer object
    :return: instantiated optimizer object
    """
    try:
        optim = getattr(torch.optim, optim_name)
    except AttributeError as err:
        print(f"Optimizer '{optim_name}' could not be found.")
        raise err.with_traceback(err.__traceback__)

    return optim(model.parameters(), lr=lr, **optimizer_kwargs)


def get_scheduler(scheduler_name, optimizer, **scheduler_kwargs):
    """
    Given a scheduler name, retrieves it from torch schedulers and instantiates it with given keyword arguments
    :param scheduler_name: the name of the loss function class, e.g. 'LinearLR', 'CosineAnnealingLR'...
    :param optimizer: optimizer object to plug the scheduler onto
    :param scheduler_kwargs: keyword arguments to the scheduler class, e.g. 'last_epoch', 'total_iters' ...
    :return: instantiated scheduler object
    """
    if scheduler_name is None:
        return None
    try:
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)
    except AttributeError as err:
        print(f"Scheduler '{scheduler_name}' could not be found.")
        raise err.with_traceback(err.__traceback__)

    return scheduler(optimizer, **scheduler_kwargs)


def get_loss(loss_name, **loss_kwargs):
    """
    Given a loss name, retrieves it from torch losses and instantiates it with given keyword arguments
    :param loss_name: the name of the loss function class, e.g. 'BCEWithLogitsLoss', 'CrossEntropyLoss'...
    :param loss_kwargs: keyword arguments to the loss function, e.g. 'pos_weight', 'reduction' ...
    :return: instantiated loss function as nn.Module object
    """
    try:
        loss_module = getattr(torch.nn.modules.loss, loss_name)
    except AttributeError as err:
        print(f"Loss '{loss_name}' could not be found.")
        raise err.with_traceback(err.__traceback__)

    return loss_module(**loss_kwargs)


def parse_kwargs_arguments(argument: str):
    """
    Returns a set of keyword arguments as a dictionary by reading a specified JSON file or parsing a JSON-like string
    :param argument: path to JSON file or JSON-like string containing keyword arguments
    :return: dictionary containing keyword arguments
    """
    if argument is None:
        kwargs = dict()
    elif os.path.isfile(argument):
        kwargs = json.load(open(argument))
    else:
        kwargs = json.loads(argument)
    return kwargs
