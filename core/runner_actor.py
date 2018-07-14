# Add the parent directory to the PATH to allow imports.
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import logging
import random
import uuid
import ray

from core.runner import DMLRunner
from custom.keras import model_from_serialized, get_optimizer
from data.iterators import count_datapoints
from data.iterators import create_train_dataset_iterator
from data.iterators import create_test_dataset_iterator
from core.utils.keras import train_keras_model, validate_keras_model
from core.utils.keras import serialize_weights


logging.basicConfig(level=logging.DEBUG,
                    format='[Runner] %(asctime)s %(levelname)s %(message)s')

@ray.remote
class DMLRunnerActor(DMLRunner):
    """
    DML Runner Actor

    This class manages the training and validation of a DML job. It trains or
    validate a specified machine learning model on a local dataset. An instance
    of this class has a one-to-one relationship with a DML job, and gets freed up
    for another DML job after executing the job. Once free, it can schedule a new
    DML job from any DML session and repeat the cycle.

    Note that this class does not schedule how jobs received should be executed
    nor listens to incoming jobs. This class only executes the jobs.

    """

    def __init__(self, *args):
        """
        Sets up the unique identifier of the DML Runner and the local dataset path.
        """
        self.runner = DMLRunner(*args)

    def send_serialized_weights(*args):
        self.runner.send_serialized_weights(*args)

    def run_job(self, *args):
        """
        Identifies a DMLJob type and executes it.

        If the runner is already executing a job, it silently does nothing.
        """
        return self.runner.run_job(*args)

    def is_active(self):
        """
        Returns whether the runner is running a job.
        """
        return self.runner.is_active()

    def _train(self, *args):
        """
        Trains the specified machine learning model on all the local data,
        starting from the initial model state specified, until a stopping
        condition is met, and using the hyper-parameters specified.

        Returns the updated model weights, the weighting factor omega, and stats
        about the training job.

        NOTE: Uses the same hyperparameters and labeler for training and
        validating during 'avg_type' of type 'val_acc'.

        NOTE2: Assumes 'initial_weights' are the actual weights and not a path.
        """
        return self.runner._train(*args)

    def _validate(self, *args):
        """
        Validates on all the local data the specified machine learning model at
        the state specified.

        Returns the metrics returned by the model.

        NOTE: Assumes 'weights' are the actual weights and not a path.
        """
        return self.runner._validate(*args)

    def _initialize_model(self, *args):
        """
        Initializes and returns the model weights as specified in the model.
        """
        return self.runner._initialize_model(*args)


