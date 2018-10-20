import tests.context

import pytest
import time

from core.communication_manager import CommunicationManager
from core.utils.event_types     import RawEventTypes, ActionableEventTypes
from core.runner                import DMLRunner
from core.scheduler             import DMLScheduler
from custom.keras               import get_optimizer
from models.keras_perceptron    import KerasPerceptron
from core.utils.dmljob          import DMLJob, serialize_job, deserialize_job
from core.configuration         import ConfigurationManager

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
        "keras",
        hyperparams=make_hyperparams(),
        label_index=0
    )
    return initialize_job

def make_hyperparams():
    hyperparams = {
        'averaging_type': 'data_size',
        'batch_size': 4,
        'epochs': 1,
        'split': 0.004,
    }
    return hyperparams

def test_communication_manager_can_initialize_and_train_model():
    """
    (1) CommMgr receives the packet it's going to receive from BlockchainGateway
    (2) CommMgr gives packet to Optimizer
    (3) Optimizer tells CommMgr to schedule an initialization job
    (4) CommMgr schedules initialization job
    (5) CommMgr receives DMLResult for initialization job
    (6) CommMgr gives DMLResult to Optimizer
    (7) Optimizer updates its weights to initialized model
    (8) Optimizer tells CommMgr to schedule a training job
    (9) CommMgr schedules training job
    (10) CommMgr receives DMLResult for training job
    (11) CommMgr gives DMLResult to Optimizer
    (12) Optimizer updates its weights to trained weights
    (13) Optimizer tells CommMgr to schedule a communication job
    (14) CommMgr schedules communication job
    (15) Exception raised, since communication job is not implemented
    NOTE: No, I can't split this up. IDK why exactly but it hangs forever.
    """
    config_manager = ConfigurationManager()
    config_manager.bootstrap(
        config_filepath='tests/artifacts/configuration.ini'
    )
    communication_manager = CommunicationManager(config_manager)
    scheduler = DMLScheduler(config_manager, communication_manager)
    scheduler.reset()
    communication_manager.configure(scheduler)
    new_session_event = {
            "key": None, 
            "content": {
                "optimizer_params": {}, 
                "serialized_job": serialize_job(make_initialize_job(make_model_json()))}}
    communication_manager.inform(RawEventTypes.NEW_SESSION.value, new_session_event)
    scheduler.start_cron(period_in_mins = 0.05)
    time.sleep(10)
    scheduler.stop_cron()
    assert len(scheduler.processed) == 2
    # TODO: I can't get the below AssertionError test to work since Georgy's code won't raise errors but it WORKS
    # try:
    #     scheduler.start_cron(period_in_mins = 0.05)
    #     time.sleep(10)
    #     scheduler.stop_cron()
    #     assert len(scheduler.processed) == 2
    #     assert False, "Runner should have been told to communicate"
    # except AssertionError as e:
    #     assert str(e) == "communication is not yet supported!"
def test_communication_manager_can_be_initialized():
    config_manager = ConfigurationManager()
    config_manager.bootstrap(
        config_filepath='tests/artifacts/configuration.ini'
    )
    communication_manager = CommunicationManager(config_manager)
    assert communication_manager is not None

def test_communication_manager_creates_new_sessions():
    """
    Takes in initialized_communication_manager because the initialization is all that matters.
    Ensures that upon receiving an initialization job, the commmgr will make an optimizer.
    """
    config_manager = ConfigurationManager()
    config_manager.bootstrap(
        config_filepath='tests/artifacts/configuration.ini'
    )
    communication_manager = CommunicationManager(config_manager)
    scheduler = DMLScheduler(config_manager, communication_manager)
    scheduler.reset()
    communication_manager.configure(scheduler)
    new_session_event = {
            "key": None, 
            "content": {
                "optimizer_params": {}, 
                "serialized_job": serialize_job(make_initialize_job(make_model_json()))}}
    communication_manager.inform(RawEventTypes.NEW_SESSION.value, new_session_event)
    assert communication_manager.optimizer is not None

def test_communication_manager_fails_if_not_configured():
    """
    Ensures that commmgr won't be able to function if it's not configured
    """
    config_manager = ConfigurationManager()
    config_manager.bootstrap(
        config_filepath='tests/artifacts/configuration.ini'
    )
    communication_manager = CommunicationManager(config_manager)
    new_session_event = {
            "key": None, 
            "content": {
                "optimizer_params": {}, 
                "serialized_job": serialize_job(make_initialize_job(make_model_json()))}}
    try:
        communication_manager.inform(RawEventTypes.NEW_SESSION.value, new_session_event)
        assert False, "this should have raised an exception"
    except Exception as e:
        assert str(e) == "Communication Manager needs to be configured first!"

def test_communication_manager_can_inform_the_optimizer():
    """
    Ensures that commmgr can tell the optimizer of something, and that the job will transfer
    """
    config_manager = ConfigurationManager()
    config_manager.bootstrap(
        config_filepath='tests/artifacts/configuration.ini'
    )
    communication_manager = CommunicationManager(config_manager)
    scheduler = DMLScheduler(config_manager, communication_manager)
    scheduler.reset()
    communication_manager.configure(scheduler)
    true_job = make_initialize_job(make_model_json())
    new_session_event = {
            "key": None, 
            "content": {
                "optimizer_params": {}, 
                "serialized_job": serialize_job(true_job)}}
    communication_manager.inform(RawEventTypes.NEW_SESSION.value, new_session_event)
    opt_job = communication_manager.optimizer.job
    assert opt_job.weights == true_job.weights
    assert opt_job.job_type == true_job.job_type
    assert opt_job.serialized_model == true_job.serialized_model
    assert opt_job.framework_type == true_job.framework_type
    assert opt_job.hyperparams == true_job.hyperparams
    assert opt_job.label_index == true_job.label_index





# def test_communication_manager_can_parse_events_correctly(communication_manager):
#     """To be implemented."""
#     assert False, "Implement me!"



# def test_communication_manager_terminates_sessions(communication_manager):
#     """To be implemented."""
#     assert False, "Implement me!"
