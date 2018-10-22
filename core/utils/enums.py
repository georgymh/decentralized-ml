"""This file contains all the different enums used to implement callback
pattern functionality."""

from enum import Enum

class JobTypes(Enum):
    """
    Enum to implement callback pattern for DMLRunner

    JOB_TRAIN: Optimizer wants Runner to train the model

    JOB_VAL: Optimizer wants Runner to validate the model

    JOB_INIT: Optimizer wants Runner to initialize the model

    JOB_AVG: Optimizer wants Runner to average weights via weighted avg.

    JOB_COMM: Optimizer wants Runner to push the weights onto
    IPFS and then communicate this to rest of nodes via blockchain
    """
    JOB_TRAIN = "TRAIN"
    JOB_VAL = "VALIDATE"
    JOB_INIT = "INITIALIZE"
    JOB_AVG = "AVERAGE"
    JOB_COMM = "COMMUNICATE"
    NOTHING = "NOTHING"

class RawEventTypes(Enum):
    """
    Enum to implement Communication Manager -> Optimizer callback pattern

    The event indicates what has occurred previously in the Unix Service.

    # NOTE: Following 3 will be replaced by MessageTypes in Gateway PR

    NEW_SESSION: Received info from the blockchain to create a new 
    decentralized learning session

    NEW_WEIGHTS: Same as above, but for new weights

    TERMINATE: Same as above, but to terminate

    JOB_DONE: Sent to Communication Manager by Scheduler, means that
    the payload will contain a DMLResult

    NEW_INFO: Sent to Communication Manager by Gateway, means that
    the payload contains info from the blockchain
    """
    NEW_SESSION = "NEW_SESSION"
    NEW_WEIGHTS = "NEW_WEIGHTS"
    TERMINATE = "TERMINATE"
    JOB_DONE = "JOB_DONE"
    NEW_INFO = "NEW_INFO"
    NOTHING = "NOTHING"

class ActionableEventTypes(Enum):
    """
    Enum to implement Optimizer -> Communication Manager callback pattern

    This is essentially a 'command' for the Comm. Mgr. so these are all imperatives.

    SCHEUDLE_JOB: Optimizer wants Communication Manager to tell its
    Scheduler to schedule a job

    TERMINATE: Optimizer wants to terminate itself

    """
    SCHEDULE_JOB = "SCHEDULE_JOB"
    TERMINATE = "TERMINATE"
    NOTHING = "NOTHING"

def callback_handler(
    callback_type,
    callback_dict,
    default=RawEventTypes.NOTHING.value
    ):
    """
    Helper function which returns a callback from a callback
    dict when given a callback type.
    """
    assert isinstance(callback_dict, dict), \
        "callback_dict was not a dict, it was: {}".format(type(callback_dict))
    callback = callback_dict[default]
    if callback_type in callback_dict:
        callback = callback_dict[callback_type]
    return callback


def callback_handler_no_default(
    callback_type,
    callback_dict,
    ):
    """
    Helper function which returns a callback from a callback
    dict when given a callback type.
    """
    assert isinstance(callback_dict, dict), \
        "callback_dict was not a dict, it was: {}".format(type(callback_dict))
    if callback_type not in callback_dict:
        raise Exception("Invalid callback passed!")
    callback = callback_dict[callback_type]
    return callback
