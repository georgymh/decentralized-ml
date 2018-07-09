import pytest
import time
import tests.context

import numpy as np
from custom.keras               import get_optimizer
from core.runner                import DMLRunner
from models.keras_perceptron    import KerasPerceptron
from core.utils.dmljob          import DMLJob, serialize_job, deserialize_job
from core.scheduler             import DMLScheduler, orchestrate_threading
from examples.labelers          import mnist_labeler

print("testing")
    


def make_dataset_path():
    return 'datasets/mnist'


def make_config():
    return {}


def make_model_json():
    m = KerasPerceptron(is_training=True)
    model_architecture = m.model.to_json()
    model_optimizer = get_optimizer(m.model)
    model_json = {
        "architecture": model_architecture,
        "optimizer": model_optimizer
    }
    return model_json


def make_hyperparams():
    hyperparams = {
        'averaging_type': 'data_size',
        'batch_size': 50,
        'epochs': 1,
        'split': 0.001,
    }
    return hyperparams


def make_initialize_job(model_json):
    initialize_job = DMLJob(
        "initialize",
        model_json,
        "keras"
    )
    return initialize_job

def make_wait_job(model_json):
    initialize_job = DMLJob(
        "initialize",
        model_json,
        "keras"
    )
    time.sleep(1)
    return initialize_job

def make_train_job(model_json, initial_weights, config, hyperparams):
    train_job = DMLJob(
        "train",
        model_json,
        "keras",
        config,
        initial_weights,
        hyperparams,
        mnist_labeler
    )
    return train_job

def test_dmlscheduler_sanity():
    model_json = make_model_json()
    scheduler = DMLScheduler(num_threads=3)
    initialize_job = make_wait_job(model_json)
    start = time.time()
    for _ in range(3):
        scheduler.add_job(initialize_job)
    can_finish = False
    total_time = 0
    count = 0
    queue = scheduler.queue
    queue.join()
    results = scheduler.results
    for thread in results:
        for initial_weights in results:
            assert type(initial_weights) == list
            assert type(initial_weights[0]) == np.ndarray

def test_dmlscheduler_speedup():
    model_json = make_model_json()
    scheduler = DMLScheduler(test=True)
    initialize_job = make_wait_job(model_json)
    start = time.time()
    for _ in range(20):
        scheduler.add_job(initialize_job)
    can_finish = False
    total_time = 0
    count = 0
    queue = scheduler.queue
    queue.join()
    total_time = time.time() - start
    assert total_time < 10


