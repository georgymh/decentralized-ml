"""This file contains all the different event types used betweent the
Communication Manager and the Optimizers."""

from enum import Enum

class ListenerEventTypes(Enum):
    """
    listener events
    """
    NEW_WEIGHTS = "new_weights"
    NOTHING = "NOTHING"

class CommMgrEventTypes(Enum):
    """
    commmgr events
    """
    NEW_SESSION = "new_session"
    NEW_WEIGHTS = "new_weights"
    SCHEDULE = "SCHEDULE"
    TERMINATE = "TERMINATE"
    NOTHING = "NOTHING"

# class RawEventTypes(Enum):
#     """
#     Dictionary of events that the Optimizer might receive from the Communication
#     Manager.

#     The event indicates what has occurred previously in the Communication Manager.

#     This is essentially a 'memory' for the Optimizer so these are all past.
#     i.e., a "TRAIN" event would contain auxiliary information about what
#     occurred, in the form of a DMLResults object.

#     Contains all the functionality of what the Communication Manager might have
#     previously done and therefore is not specific to any one Optimizer.
#     """
#     SCHEDULE = "SCHEDULE"
#     TERMINATE = "TERMINATE"
#     NOTHING = "NOTHING"

class ActionableEventTypes(Enum):
    """
    Dictionary of events that the Communication Manager might receive from the
    Optimizer.

    The event indicates what state the Optimizer has transitioned to. This is
    essentially a 'command' for the Comm. Mgr. so these are all imperatives.

    This dictionary has to contain all possible states that an Optimizer can be in.
    """
    TRAIN = "TRAIN"
    COMMUNICATE = "COMMUNICATE"
    AVERAGING = "AVERAGING"
    SPLIT = "SPLIT"
    TERMINATE = "TERMINATE"
    # WEIGHT = "WEIGHT"
    NOTHING = "NOTHING"
