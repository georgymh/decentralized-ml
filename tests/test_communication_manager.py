import tests.context

import pytest

from core.configuration import ConfigurationManager
from core.communication_manager import CommunicationManager


@pytest.fixture
def config_manager():
    config_manager = ConfigurationManager()
    config_manager.bootstrap(
        config_filepath='tests/artifacts/configuration.ini'
    )
    return config_manager


def test_communication_manager_can_be_initialized(config_manager):
    communication_manager = CommunicationManager(config_manager=config_manager)
    assert communication_manager is not None

def test_communication_manager_creates_new_sessions(config_manager):
    """To be implemented."""
    assert False, "Implement me!"

def test_communication_manager_fails_if_not_configured(config_manager):
    """To be implemented."""
    assert False, "Implement me!"

def test_communication_manager_can_inform_the_optimizer(config_manager):
    """To be implemented."""
    assert False, "Implement me!"

def test_communication_manager_can_parse_events_correctly(config_manager):
    """To be implemented."""
    assert False, "Implement me!"

def test_communication_manager_can_schedule_jobs(config_manager):
    """To be implemented."""
    assert False, "Implement me!"

def test_communication_manager_terminates_sessions(config_manager):
    """To be implemented."""
    assert False, "Implement me!"
