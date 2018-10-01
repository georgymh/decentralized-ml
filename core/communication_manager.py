import logging
from core.EventTypes import CommMgrEventTypes, OptimizerEventTypes
from core.fedavgoptimizer import FederatedAveragingOptimizer
# TODO: Need to import 'FederatedAveragingOptimizer' (@panda).


logging.basicConfig(level=logging.DEBUG,
    format='[CommunicationManager] %(asctime)s %(levelname)s %(message)s')


class CommunicationManager(object):
    """
    Communication Manager

    Manages the communication within the modules of the service.

    Right now, the service can only have one session at a time. This will be
    reflected in the code by just having one optimizer property and no
    information about sessions throughout the code.

    """

    def __init__(self, config_manager):
        """
        Initializes the class.
        """
        logging.info("Setting up Communication Manager...")
        self.config_manager = config_manager
        self.optimizer = None # NOTE: Right now we're only dealing with one
                              # session at a time, so the class is only
                              # concerned with one optimizer.
        self.scheduler = None
        logging.info("Communication Manager is set up!")

    def configure(self, scheduler):
        """
        Configures the Communication Manager so that it can submit jobs to the
        Scheduler.
        """
        logging.info("Configuring the Communication Manager...")
        self.scheduler = scheduler
        logging.info("Communication Manager is configured!")

    def create_session(self, session_id, serialized_job, network, optimizer_params):
        """
        Creates a new session and optimizer based on the parameters sent by a
        model developer. It then asks the optimizer what's the first thing to do
        and the service starts working on that session from that point on.
        """
        # NOTE: We removed the 'optimizer_type' argument since for the MVP we're
        # only considering the 'FederatedAveragingOptimizer' for now.
        self.optimizer = FederatedAveragingOptimizer(optimizer_params)
        actionable_event = self.optimizer.kickoff()
        self.parse_and_run_callback(actionable_event)

    def inform(self, event_type, payload):
        """
        Method called by other modules to inform the Communication Manager about
        events that are going on in the service.

        These payloads are relayed to the active optimizer (right now the only
        one, in the future, the one corresponding to the session_id passed),
        which decides how and to whom the Communication Manager should
        communicate the event/message.

        For example: A runner could inform the Communication Manager that the
        node is done training a particular model, to which the Optimizer could
        decide it's time to communicate the new weights to the network.
        """
        event = {
            "event_type": event_type,
            "payload": payload
        }
        actionable_event = self.optimizer.ask(event)
        self.parse_and_run_callback(actionable_event)

    def _parse_and_run_callback(self, actionable_event):
        """
        Parses an actionable_event returned by an optimizer and runs the
        corresponding "callback" based on the event type, which could be to do
        nothing.

        The way this method parses an event is by stripping out the event type
        and sending the raw payload to the callback function, which will handle
        everything from there on.
        """
        event_type = actionable_event.get(
            'optimizer_state',
            CommMgrEventTypes.NOTHING.value
        )
        callback = EVENT_TYPE_CALLBACKS[CommMgrEventTypes.NOTHING.value]
        if event_type in EVENT_TYPE_CALLBACKS:
            callback = EVENT_TYPE_CALLBACKS[event_type]
        payload = actionable_event.get('payload', None)
        callback(payload)

    def _schedule_job(self, payload):
        """
        Helper function that creates a DMLJob based on the given payload and
        schedules the job through the Scheduler.
        """
        if not self.scheduler:
            raise Exception("Communication Manager needs to be configured first!")
        if not payload:
            raise Exception("Payload is not valid!")
        job = self.deserialize_job(payload)
        self.scheduler.add_job(job)

    def _terminate_session(self, payload):
        """
        Helper function that removes the current session (and optimizer) from
        the Communication Manager.

        Note that it ignores the payload for now (since we're only dealing with
        one session).
        """
        self.optimizer = None

    EVENT_TYPE_CALLBACKS = {
        OptimizerEventTypes.TRAIN.value: _schedule_job,
        OptimizerEventTypes.COMM.value: _schedule_job,
		OptimizerEventTypes.AVERAGE.value: _schedule_job,
        OptimizerEventTypes.SPLIT.value: _schedule_job,
		OptimizerEventTypes.TERMINATE.value: _terminate_session,
        OptimizerEventTypes.NOTHING.value: lambda *args: None
    }
