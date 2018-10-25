import tests.context

import pytest

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

def test_optimize_receives_result_of_training():
    optimizer = FederatedAveragingOptimizer(initialization_packet)
    # give optimizer DMLResults
    model_json = make_model_json()
    runner = DMLRunner(config_manager)
    initialize_job = make_initialize_job(model_json)
    result = runner.run_job(initialize_job)
    results = result.results
    initial_weights = results['weights']
    optimizer.ask(RawEventTypes.JOB_DONE, result)
    # make sure its job is updated
    assert optimizer.job.weights == initial_weights