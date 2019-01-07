import pytest
import multiprocessing
import time
from multiprocessing import Manager
from multiprocessing import Pool, Queue

class MockRunner(object):
    def __init__(self):
        self.dummy = "hello world"
    def run_job(self, data):
        data.append(self.dummy)

@pytest.fixture
def runner():
    return MockRunner()

@pytest.fixture
def state():
    return Manager().list()

@pytest.fixture
def gateway():
    class MockGateway(object):
        def __init__(self):
            self.state = Manager().list()
    return MockGateway()

@pytest.fixture
def scheduler(gateway):
    class MockScheduler(object):
        def __init__(self):
            self.processed = []
            self.history = []
            self.num_runners = 1
            self.max_tries = 1
            self.queue = Queue(self.num_runners * self.num_runners)
            multiprocessing.set_start_method('spawn', force=True)
            self.pool = Pool(processes=self.num_runners)
            self.runners = [MockRunner() for _ in range(self.num_runners)]
            self.current_jobs = [None for _ in range(self.num_runners)]
            self.current_results = [None for _ in range(self.num_runners)]
        def configure(self, gateway):
            self._blockchain_state = gateway.state
        def add_job(self, dml_job):
            """
            Add a job to the queue.
            """
            self.queue.put(dml_job)
        def runners_run_next_jobs(self):
            """
            Check each job to see if it has a job running. If not, have the runner
            run the next job on the queue asynchronously of the others and collect
            the result of the job that was running before (if applicable).

            If job succeeds, inform the Communication Manager about the results.

            If job that runner is running fails, then put the job back in queue.
            If job failed more than num_tries times, don't put the job back in
            the queue.
            """
            for i, runner in enumerate(self.runners):
                # Check if there's any finished jobs and process them.
                if self.current_results[i]:
                    #If the job results are available
                    finished_results = self.current_results[i].get()
                    # If the job finished successfully...
                    self.processed.append(finished_results)
                    self.history.append(finished_results)
                    self.current_jobs[i] = None
                    self.current_results[i] = None
                # Check if there's any queued jobs and schedule them.
                if not self.queue.empty():
                    # If there's something to be scheduled...
                    job_to_run = self.queue.get(block=False)
                    self.current_jobs[i] = job_to_run
                    self.current_results[i] = self.pool.apply_async(
                        runner.run_job,
                        (self._blockchain_state,)
                    )
    scheduler = MockScheduler()
    scheduler.configure(gateway)
    return scheduler

def test_state(gateway):
    assert type(gateway.state) is not list

def test_gateway_setting_scheduler(gateway, scheduler):
    scheduler.add_job("hello")
    scheduler.runners_run_next_jobs()
    timeout = time.time() + 3
    while time.time() < timeout and not scheduler.processed:
        time.sleep(0.1)
        scheduler.runners_run_next_jobs()
    assert gateway.state[0] == "hello world"
