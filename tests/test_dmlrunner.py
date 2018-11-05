import tests.context

import pytest
import numpy as np
import pandas as pd
import os
import shutil

from core.runner                import DMLRunner
from core.configuration         import ConfigurationManager
from core.utils.keras           import serialize_weights
from core.utils.enums           import JobTypes
from tests.testing_utils        import make_initialize_job, make_model_json
from tests.testing_utils        import make_train_job, make_validate_job, make_hyperparams
from tests.testing_utils        import make_transform_split_job
                                       

@pytest.fixture
def config_manager():
    config_manager = ConfigurationManager()
    config_manager.bootstrap(
        config_filepath='tests/artifacts/runner_scheduler/configuration.ini'
    )
    return config_manager


@pytest.fixture
def sample_transform_function():
    return lambda x: x.drop_duplicates()

@pytest.fixture
def mnist_filepath():
    return 'tests/artifacts/runner_scheduler/mnist'

@pytest.fixture
def small_filepath():
    return 'tests/artifacts/runner_scheduler/test'

def accurate_transform_test(dsm, expected_test_transformed, split):
    """
    Check that the data in this folder is the result of calling the transform function on
    each csv in the raw data filepath
    """
    transform_dsm = dsm.get_transformed_data()
    keys = list(transform_dsm.keys())
    train = transform_dsm['train']
    test = transform_dsm['test']
    count = len(expected_test_transformed)
    assert len(train) == int(split * count)
    assert len(test) == int((1 - split) * count)
    actual_test_transformed = train.append(test)
    actual_test_transformed = actual_test_transformed.reset_index(drop=True)
    merged_data = pd.merge(
        actual_test_transformed,
        expected_test_transformed,
        on = ['a', 'b', 'c'],
        how = 'inner'
    )
    assert len(actual_test_transformed) == count
    assert len(merged_data) == count

def test_dmlrunner_initialize_job_returns_list_of_nparray(config_manager):
    model_json = make_model_json()
    runner = DMLRunner(config_manager)
    initialize_job = make_initialize_job(model_json)
    result = runner.run_job(initialize_job)
    results = result.results
    initial_weights = results['weights']
    assert result.job.job_type is JobTypes.JOB_INIT.name
    assert type(initial_weights) == list
    assert type(initial_weights[0]) == np.ndarray

def test_dmlrunner_train_test_split( \
        config_manager, small_filepath, sample_transform_function):
    model_json = make_model_json()
    runner = DMLRunner(config_manager)
    transform_split_job = make_transform_split_job(
                            model_json, 
                            small_filepath, 
                            sample_transform_function
                        )
    transform_split_job.hyperparams['split'] = 0.75
    job_results = runner.run_job(transform_split_job)
    session_filepath = job_results.results['session_filepath']
    assert os.path.isdir(session_filepath), \
        "Session folder does not exist!"
    train_filepath = os.path.join(session_filepath, 'train.csv')
    test_filepath = os.path.join(session_filepath, 'test.csv')
    assert os.path.isfile(train_filepath) and os.path.isfile(test_filepath), \
        "Training and test set not created!"
    train = pd.read_csv(train_filepath)
    test = pd.read_csv(test_filepath)
    assert len(train) + len(test) == 4, \
        "Transform function was not applied correctly."
    assert len(train) == 3 and len(test) == 1, \
        "Train test split was not performed correctly."

    # Clean up
    if os.path.isdir(session_filepath):
        shutil.rmtree(session_filepath)

def test_dmlrunner_train_job_returns_weights_omega_and_stats( \
        config_manager, mnist_filepath):
    model_json = make_model_json()
    hyperparams = make_hyperparams()
    runner = DMLRunner(config_manager)
    initialize_job = make_initialize_job(model_json)
    initial_weights = runner.run_job(initialize_job).results['weights']
    transform_split_job = make_transform_split_job(
                            model_json, 
                            mnist_filepath
                        )
    job_results = runner.run_job(transform_split_job)
    session_filepath = job_results.results['session_filepath']
    train_job = make_train_job(
                    model_json, 
                    initial_weights, 
                    hyperparams, 
                    session_filepath
                )
    result = runner.run_job(train_job)
    results = result.results
    new_weights = results['weights']
    omega = results['omega']
    train_stats = results['train_stats']
    assert result.job.job_type is JobTypes.JOB_TRAIN.name
    assert type(new_weights) == list
    assert type(new_weights[0]) == np.ndarray
    assert type(omega) == int or type(omega) == float
    assert type(train_stats) == dict

    # Clean up
    if os.path.isdir(session_filepath):
        shutil.rmtree(session_filepath)
        


def test_dmlrunner_validate_job_returns_stats( \
        config_manager, mnist_filepath):
    model_json = make_model_json()
    hyperparams = make_hyperparams()
    runner = DMLRunner(config_manager)
    initialize_job = make_initialize_job(make_model_json())
    initial_weights = runner.run_job(initialize_job).results['weights']
    transform_split_job = make_transform_split_job(
                            model_json, 
                            mnist_filepath
                        )
    job_results = runner.run_job(transform_split_job)
    session_filepath = job_results.results['session_filepath']
    train_job = make_train_job(
                    model_json, 
                    initial_weights, 
                    hyperparams, 
                    session_filepath
                )
    result = runner.run_job(train_job)
    results = result.results
    new_weights = results['weights']
    omega = results['omega']
    train_stats = results['train_stats']
    hyperparams['split'] = 1 - hyperparams['split']
    validate_job = make_validate_job(model_json, 
                    new_weights, 
                    hyperparams, 
                    session_filepath
                )
    result = runner.run_job(validate_job)
    results = result.results
    val_stats = results['val_stats']
    assert result.job.job_type is JobTypes.JOB_VAL.name
    assert type(val_stats) == dict

    # Clean up
    if os.path.isdir(session_filepath):
        shutil.rmtree(session_filepath)
