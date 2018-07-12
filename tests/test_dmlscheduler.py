import pytest
import time
import tests.context

import logging
import numpy as np
from custom.keras               import get_optimizer
from core.runner                import DMLRunner
from core.runner_actor          import DMLRunnerActor
from models.keras_perceptron    import KerasPerceptron
from core.utils.dmljob          import DMLJob, serialize_job, deserialize_job
from core.scheduler             import DMLScheduler
from examples.labelers          import mnist_labeler
from queue import Queue
print("testing")


import ray


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

scheduler = DMLScheduler()

def test_dmlrunner_initialize_job_returns_list_of_nparray():
    model_json = make_model_json()
    runner = DMLRunner(make_dataset_path(), make_config())
    initialize_job = make_initialize_job(model_json)
    initial_weights = runner.run_job(initialize_job)
    assert type(initial_weights) == list
    assert type(initial_weights[0]) == np.ndarray

def ahh_test_2():
    model_json = make_model_json()
    runner = DMLRunnerActor.remote(make_dataset_path(), make_config())
    initialize_job = make_initialize_job(model_json)
    object_id = runner.run_job.remote(initialize_job)
    initial_weights = ray.get(object_id)
    assert type(initial_weights) == list
    assert type(initial_weights[0]) == np.ndarray

def test_dmlscheduler_sanity():
    model_json = make_model_json()
    initialize_job = make_initialize_job(model_json)
    scheduler.add_job(initialize_job)
    scheduler._runners_run_next_jobs()
    start = time.time()
    while not scheduler.processed:
        scheduler._runners_run_next_jobs()
    initial_weights = scheduler.processed.pop(0)
    print(time.time()-start)
    assert type(initial_weights) == list
    assert type(initial_weights[0]) == np.ndarray
    scheduler.reset()

def test_dmlscheduler_speedup_naive():
    scheduler.reset()
    slow = 0
    for _ in range(10):
        model_json = make_model_json()
        initialize_job = make_initialize_job(model_json)
        scheduler.add_job(initialize_job)
        start = time.time()
        while len(scheduler.processed) < 1:
            scheduler._runners_run_next_jobs()
        slow += time.time() - start
        scheduler.reset()

    for _ in range(10):
        model_json = make_model_json()
        initialize_job = make_initialize_job(model_json)
        scheduler.add_job(initialize_job)
    start = time.time()
    while len(scheduler.processed) < 10:
        scheduler._runners_run_next_jobs()
    fast = time.time() - start
    print(slow)
    print(fast)
    assert fast*2 < slow
    scheduler.reset()


