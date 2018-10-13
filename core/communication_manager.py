import logging
from core.utils.event_types import RawEventTypes, ActionableEventTypes
#from core.fedavgoptimizer import FederatedAveragingOptimizer
# TODO: Need to import 'FederatedAveragingOptimizer' (@panda).


logging.basicConfig(level=logging.DEBUG,
    format='[CommunicationManager] %(asctime)s %(levelname)s %(message)s')


class CommunicationManager(object):
    """
    Communication Manager

    Manages the communication within the modules of the service throughout all
    active DML Sessions.

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
        self.optimizer = None # NOTE: Since right now we're only dealing with
                              # one session at a time, the class is only
                              # concerned with one optimizer.
        self.scheduler = None
        logging.info("Communication Manager is set up!")

    def configure(self, scheduler, listener):
        """
        Configures the Communication Manager so that it can submit jobs to the
        Scheduler and Listener.
        """
        logging.info("Configuring the Communication Manager...")
        self.scheduler = scheduler
        self.listener = listener
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
        self._parse_and_run_callback(actionable_event)

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
            "raw_event_type": event_type,
            "payload": payload,
        }
        actionable_event = self.optimizer.ask(event)
        self._parse_and_run_callback(actionable_event)

    def _parse_and_run_callback(self, actionable_event):
        """
        Parses an actionable_event returned by an optimizer and runs the
        corresponding "callback" based on the event type, which could be to do
        nothing.

        The way this method parses an event is by stripping out the event type
        and sending the processed payload (which should be a DML Job) to the
        callback function, which will handle everything from there on.
        """
        event_type = actionable_event.get(
            'actionable_event_type',
            ActionableEventTypes.NOTHING.value
        )
        callback = ACTIONABLE_EVENT_TYPE_2_CALLBACK.get(
            event_type,
            ActionableEventTypes.NOTHING.value
        )
        job = actionable_event.get('job', None)
        callback(job)

    def _schedule_job(self, dmljob_obj):
        """
        Helper function that schedules a DML Job through the Scheduler.
        """
        if not self.scheduler:
            raise Exception("Communication Manager needs to be configured first!")
        if not dmljob_obj:
            raise Exception("Job is not valid!")
        self.scheduler.add_job(dmljob_obj)

    def _communication_job(self, payload):
        """
        Helper function that tells the Listener what to upload.
        """
        self.listener.inform("new_weights", payload)

    def _terminate_session(self, dummy_param):
        """
        Helper function that removes the current session (and optimizer) from
        the Communication Manager.

        Note that it ignores the payload for now (since we're only dealing with
        one session).
        """
        self.optimizer = None

    ACTIONABLE_EVENT_TYPE_2_CALLBACK = {
        ActionableEventTypes.TRAIN.value: _schedule_job,
        ActionableEventTypes.COMMUNICATE.value: _communication_job,
		ActionableEventTypes.AVERAGE.value: _schedule_job,
        ActionableEventTypes.SPLIT.value: _schedule_job,
		ActionableEventTypes.TERMINATE.value: _terminate_session,
        ActionableEventTypes.NOTHING.value: lambda *args: None
    }
