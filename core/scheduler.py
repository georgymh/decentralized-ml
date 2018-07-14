import logging
import time
import json
from collections import deque

import ray
import schedule

from core.utils.dmljob import DMLJob
from core.runner_actor import DMLRunnerActor


ray.init()
logging.basicConfig(level=logging.DEBUG,
                    format='[Scheduler] %(asctime)s %(levelname)s %(message)s')

class DMLScheduler(object):
    """
    DML Scheduler
    This class schedules and manages the execution of DMLJobs using the DMLRunner.
    Note: now supports a multithreaded environment using Ray.
    Note2: only supports one dataset type.
    Note3: Singleton.
    """
    # Here will be the instance stored.
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if DMLScheduler.__instance == None:
            DMLScheduler()
        return DMLScheduler.__instance

    #Default number of runners that are running at a time is 4
    def __init__(self, num_runners=4):
        """ Virtually private constructor. """
        self.num_runners = num_runners
        if DMLScheduler.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DMLScheduler.__instance = self
        logging.info("Setting up scheduler...")
        self.queue = deque()
        self.object_ids = [None]*4
        self.processed = []
        with open('core/config.json') as f:
            config = json.load(f)
            self.dataset_path = config["dataset_path"]
            self.runner_config = config["runner_config"]
            scheduler_config = config["scheduler_config"]
            run_frequency_mins = scheduler_config["frequency_in_mins"]
            num_processes = scheduler_config["num_processes"]
        self.runners = [self.create_runner() for _ in range(self.num_runners)]
        #self.threading_pool = ThreadPool(num_processes)
        self._start_cron(run_frequency_mins)
        logging.info("Scheduler is set up!")

    def add_job(self, dml_job):
        """ Add a job to the queue"""
        assert type(dml_job) is DMLJob, "Job is not of type DMLJob."
        logging.info("Scheduling job...")
        self.queue.append(dml_job) 

    def create_runner(self):
        """ Creates new runner. """
        return DMLRunnerActor.remote(self.dataset_path, self.runner_config)

    #
    def _runners_run_next_jobs(self):
        """ Check each job to see if it has a job running. If not, have the runner run the 
        next job on the queue asynchronously of the others and collect the result of the job
        that was running before (if applicable) """
        for i in range(self.num_runners):
            if not ray.get(self.runners[i].is_active.remote()):
                if self.object_ids[i]:
                    self.processed.append(ray.get(self.object_ids[i]))
                    self.object_ids[i] = None
                if self.queue:
                    self.object_ids[i] = self.runners[i].run_job.remote(self.queue.popleft())

    def _start_cron(self, run_frequency_mins):
        """ CRON job to run next jobs on runners, if applicable. """
        logging.info("Starting cron...")
        schedule.every(run_frequency_mins).minutes.do(self._runners_run_next_jobs)
        schedule.run_pending()
        logging.info("Cron started!")

    #
    def reset(self):
        """scheduler is a singleton, so need to get a new instance inplace """
        self.queue = deque()
        self.processed = []

    # def __getstate__(self):
    #     self_dict = self.__dict__.copy()
    #     del self_dict['threading_pool']
    #     return self_dict
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

if __name__ == '__main__':
    # Set up model
    from models.keras_perceptron import KerasPerceptron
    from custom.keras import get_optimizer
    m = KerasPerceptron(is_training=True)
    model_architecture = m.model.to_json()
    model_optimizer = get_optimizer(m.model)
    model_json = {
        "architecture": model_architecture,
        "optimizer": model_optimizer
    }
    print(model_json)

    # Set up hyperparams
    from examples.labelers import mnist_labeler
    config = {}
    hyperparams = {
        'averaging_type': 'data_size',
        'batch_size': 50,
        'epochs': 1,
        'split': 0.8,
    }

    # Schedule some jobs
    from core.utils.dmljob import DMLJob

    scheduler = DMLScheduler.getInstance()

    initialize_job = DMLJob(
        "initialize",
        model_json,
        "keras"
    )
    initial_weights = scheduler.add_job(initialize_job)

    train_job = DMLJob(
        "train",
        model_json,
        "keras",
        config,
        initial_weights,
        hyperparams,
        mnist_labeler
    )
    new_weights, _, _ = scheduler.add_job(train_job)

    validate_job = DMLJob(
        "validate",
        model_json,
        'keras',
        config,
        new_weights,
        hyperparams,
        mnist_labeler
    )
    scheduler.add_job(validate_job)