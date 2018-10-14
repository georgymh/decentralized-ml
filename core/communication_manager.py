import logging
from core.utils.event_types import CommMgrEventTypes, ActionableEventTypes
from core.fedavgoptimizer import FederatedAveragingOptimizer
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
        self.ACTIONABLE_EVENT_TYPE_2_CALLBACK = {
            ActionableEventTypes.TRAIN.value: self._schedule_job,
            ActionableEventTypes.AVERAGE.value: self._schedule_job,
            ActionableEventTypes.COMMUNICATE.value: self._communication_job,
            ActionableEventTypes.SPLIT.value: self._transform_and_split,
            ActionableEventTypes.TERMINATE.value: self._terminate_session,
            ActionableEventTypes.NOTHING.value: self._do_nothing,
        }

        # These are unmapped and will only be received when an optimizer is not
        # active.
        self.RAW_EVENT_TYPE_2_CALLBACK = {
            RawEventTypes.NEW_SESSION.value: self._create_session,
            RawEventTypes.NEW_WEIGHTS.value: self._do_nothing,
            RawEventTypes.TERMINATE.value: self._terminate_session,

        }

    def configure(self, scheduler, listener):
        """
        Configures the Communication Manager so that it can submit jobs to the
        Scheduler and Listener.
        """
        logging.info("Configuring the Communication Manager...")
        self.scheduler = scheduler
        self.listener = listener
        logging.info("Communication Manager is configured!")

    def __configure_listener(self, listener):
        """
        NOTE: ONLY USED FOR THE BLOCKCHAIN BRANCH.

        Dependency injection for Listener.
        """
        self.listener = listener

    def inform(self, event_type, payload):
        """
        Method called by other modules to inform the Communication Manager about
        events that are going on in the service.

        If there isn't an active optimizer, the commmgr will handle it
        (make a new optimizer if the call is toto make a new optimizer) or
        do nothing (if the call wasn't to make a new optimizer)

        Else, payloads are relayed to the active optimizer (right now the only
        one, in the future, the one corresponding to the session_id passed),
        which decides how and to whom the Communication Manager should
        communicate the event/message.

        For example: A runner could inform the Communication Manager that the
        node is done training a particular model, to which the Optimizer could
        decide it's time to communicate the new weights to the network.
        """
        logging.info("Information has been received: type({})".format(event_type))
        raw_event = {
            "raw_event_type": event_type,
            "payload": payload
        }
        if self.optimizer:
            logging.info("Optimizer available. Asking optimizer...")
            actionable_event = self.optimizer.ask(raw_event)
            self._parse_and_run_callback(actionable_event)
        else:
            # This happens when we need to start the session for the first time.
            # We're passing a raw event.
            logging.info("No optimizer available. This should be a create session.")
            self._parse_and_run_raw_event(raw_event)


    # Parsing routines.

    def _parse_and_run_raw_event(self, self_event):
        """
        Parses a raw_event given by the listener and runs the corresponding
        "callback" based on the event type, which could be nothing.
        """
        event_type = actionable_event.get('raw_event_type', None)
        callback = self.ACTIONABLE_EVENT_TYPE_2_CALLBACK.get(
            event_type,
            ActionableEventTypes.NOTHING.value
        )
        payload = self_event.get('payload', None)
        callback(payload)

    def _parse_and_run_callback(self, actionable_event):
        """
        Parses an actionable_event returned by an optimizer and runs the
        corresponding "callback" based on the event type, which could be to do
        nothing.

        The way this method parses an event is by stripping out the event type
        and sending the processed payload (which should be a DML Job) to the
        callback function, which will handle everything from there on.
        """
        event_type = actionable_event.get('actionable_event_type', None)
        callback = self.ACTIONABLE_EVENT_TYPE_2_CALLBACK.get(
            event_type,
            ActionableEventTypes.NOTHING.value
        )
        job = actionable_event.get('job', None)
        callback(job)


    # Callbacks.

    def _create_session(self, payload):
        """
        Creates a new session and optimizer based on the parameters sent by a
        model developer. It then asks the optimizer what's the first thing to do
        and the service starts working on that session from that point on.
        """
        # NOTE: We removed the 'optimizer_type' argument since for the MVP we're
        # only considering the 'FederatedAveragingOptimizer' for now.
        # TODO: We need to incorporate session id's when we're ready.
        logging.info("New optimizer session is being set up: {}".format(payload))
        session_id = payload.get('key')
        content = payload.get('content')

        optimizer_params = content.get('optimizer_params')
        serialized_job = content.get('serialized_job')

        self.optimizer = FederatedAveragingOptimizer(content)
        actionable_event = self.optimizer.kickoff()
        self._parse_and_run_callback(actionable_event)

    def _transform_and_split(self, payload):
        """
        Transforms and splits the local dataset.

        TO BE IMPLEMENTED.
        """
        pass

    def _schedule_job(self, payload):
        """
        Helper function that schedules a DML Job through the Scheduler.
        """
        if not self.scheduler:
            raise Exception("Communication Manager needs to be configured first!")
        if not payload and not payload.get('job', None):
            raise Exception("Payload is not valid!")
        dmljob_obj = deserialize_job(payload['job'])
        if not dmljob_obj:
            raise Exception("Job is not valid!")
        self.scheduler.add_job(dmljob_obj)

    def _communication_job(self, payload):
        """
        Helper function that tells the Listener what to upload.
        """
        self.listener.inform("new_weights", payload)

    def _terminate_session(self, payload):
        """
        Helper function that removes the current session (and optimizer) from
        the Communication Manager.

        Note that it ignores the payload for now (since we're only dealing with
        one session).
        """
        self.optimizer = None

    def _do_nothing(self, payload):
        """
        Do nothing.
        """
        logging.info("Doing nothing.")
        pass
