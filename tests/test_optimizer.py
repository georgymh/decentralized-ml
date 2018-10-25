import tests.context

import pytest
import numpy as np

from tests.testing_utils import *
from core.utils.enums import *
from core.fed_avg_optimizer import FederatedAveragingOptimizer
from core.runner import DMLRunner
from core.configuration import ConfigurationManager
@pytest.fixture
def config_manager():
    config_manager = ConfigurationManager()
    config_manager.bootstrap(
        config_filepath='tests/artifacts/runner_scheduler/configuration.ini'
    )
    return config_manager
initialization_packet = {
    "optimizer_params": {},
    "serialized_job": make_serialized_job()
}

def test_optimizer_can_kickoff():
    optimizer = FederatedAveragingOptimizer(initialization_packet)
    assert optimizer.kickoff()[0] == ActionableEventTypes.SCHEDULE_JOB.name

def test_optimizer_updates_weights_as_result_of_job(config_manager):
    optimizer = FederatedAveragingOptimizer(initialization_packet)
    # give optimizer DMLResults
    model_json = make_model_json()
    runner = DMLRunner(config_manager)
    initialize_job = make_initialize_job(model_json)
    result = runner.run_job(initialize_job)
    results = result.results
    initial_weights = results['weights']
    optimizer.ask(RawEventTypes.JOB_DONE.name, result)
    # make sure its job is updated
    assert all([np.count_nonzero(arr)==0 for arr in np.subtract(optimizer.job.weights, initial_weights)])

def test_optimizer_schedules_training_after_initialization(config_manager):
    optimizer = FederatedAveragingOptimizer(initialization_packet)
    model_json = make_model_json()
    runner = DMLRunner(config_manager)
    initialize_job = make_initialize_job(model_json)
    result = runner.run_job(initialize_job)
    retval = optimizer.ask(RawEventTypes.JOB_DONE.name, result)
    assert retval[0] == ActionableEventTypes.SCHEDULE_JOB.name
    assert retval[1].job_type == JobTypes.JOB_TRAIN.name

def test_optimizer_schedules_communication_after_training(config_manager):
    optimizer = FederatedAveragingOptimizer(initialization_packet)
    model_json = make_model_json()
    runner = DMLRunner(config_manager)
    initialize_job = make_initialize_job(model_json)
    result = runner.run_job(initialize_job)
    results = result.results
    initial_weights = results['weights']
    optimizer.ask(RawEventTypes.JOB_DONE.name, result)
    train_job = make_train_job(model_json, initial_weights, make_hyperparams())
    result = runner.run_job(train_job)
    results = result.results
    trained_weights = results['weights']
    retval = optimizer.ask(RawEventTypes.JOB_DONE.name, result)
    assert all([np.count_nonzero(arr)==0 for arr in np.subtract(optimizer.job.weights, trained_weights)])
    assert retval[0] == ActionableEventTypes.SCHEDULE_JOB.name
    assert retval[1].job_type == JobTypes.JOB_COMM.name