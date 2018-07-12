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


def make_initialize_job(model_json):
    initialize_job = DMLJob(
        "initialize",
        model_json,
        "keras"
    )
    return initialize_job

scheduler = DMLScheduler()

#Check that the scheduling/running functionality is maintained.
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

#Check that running 10 jobs with Ray takes at most half the 
# time to run them serially (i.e. there is a speedup)
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
    assert fast*1.5 < slow
    scheduler.reset()


