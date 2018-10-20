from core.utils.event_types import ActionableEventTypes, RawEventTypes, callback_handler
from core.utils.dmljob import deserialize_job
from core.utils.dmlresult import DMLResult

class FederatedAveragingOptimizer(object):
	"""
	Federated Averaging Optimizer

	Implements the Federated Averaging algorithm (with some tweaks)
	and acts as a "brain" for the Unix Service

	"""
	def __init__(self, content):
		"""
		Kwargs is a dictionary containing all the info needed to initialize the Optimizer.
		This includes any number of properties such as thresholds, etc.
		"""
		kwargs = content.get('optimizer_params')
		serialized_job = content.get('serialized_job')
		# The Optimizer knows its job
		self.job = deserialize_job(serialized_job)
		self.listen_iterations = kwargs.get('listen_iterations', 0)
		self.listen_bound = kwargs.get('listen_bound', 1)
		self.LEVEL_1_CALLBACKS = {
			RawEventTypes.JOB_DONE.value: self.handle_job_done,
			RawEventTypes.NEW_INFO.value: self.handle_new_info,
			RawEventTypes.NOTHING.value: self.do_nothing,
		}
		self.LEVEL_2_JOB_CALLBACKS = {
			RawEventTypes.JOB_TRAIN.value: self._done_training,
			RawEventTypes.JOB_INIT.value: self._done_initializing,
			RawEventTypes.JOB_AVG.value: self._average_iter,
			RawEventTypes.NOTHING.value: self.do_nothing,
		}
		self.LEVEL_2_INFO_CALLBACKS = {
			RawEventTypes.NEW_WEIGHTS.value: self._handle_new_weights,
			RawEventTypes.TERMINATE.value: self._done_everything,
			RawEventTypes.NOTHING.value: self.do_nothing,
		}

	def handle_job_done(self, result):
		"""
		Level 1 Callback to handle the payload of a DMLResult
		"""
		# assert payload is a DMLResult
		assert isinstance(result, DMLResult)
		# assume success for now
		# get the DMLJob from the DMLResult
		new_weights = result.results.get('weights')
		self._update_weights(new_weights)
		job_type = result.job.job_type
		callback = callback_handler(job_type, self.LEVEL_2_JOB_CALLBACKS)
		return callback(result)

	def handle_new_info(self, payload):
		"""
		Level 1 Callback to handle the info from the network
		Case 1: New weights
		Case 2: Termination
		Case 3: Init ???
		"""
		# event_type = payload['info_type']
		# callback = callback_handler(event_type, self.LEVEL_2_INFO_CALLBACKS)
		# NOTE: Hardcoded to handle new weights for now...
		callback = callback_handler("new_weights", self.LEVEL_2_INFO_CALLBACKS)
		return callback(payload)

	def _update_weights(self, weights):
		"""
		Helper function to replace the weights in the optimizer's
		currently stored DMLJob with these weights, therefore 
		ensuring that any future DMLJobs will operate with the 
		correct weights. Mutates, does not return anything.
		Case 1: Initialization
		Case 2: Training update
		Case 3: Averaging update
		"""
		self.job.weights = weights

	def _handle_new_weights(self, payload):
		"""
		Level 2 Callback to handle new weights info
		Updates state and then returns appropriate DMLJob
		"""
		job = self.job
		job.job_type = "average"
		serialized_weights = payload["content"]
		job.add_weights(serialized_weights)
		return ActionableEventTypes.SCHEDULE_JOB.value, job
	
	def _done_training(self, payload):
		"""
		Level 2 Callback to handle a DMLResult of a training job
		Should return a DMLJob of type Communication
		"""
		job = self.job
		job.job_type = "communicate"
		return ActionableEventTypes.SCHEDULE_JOB.value, job
	
	def _done_initializing(self, payload):
		"""
		Level 2 Callback to handle a DMLResult of an initialization job
		Should return a DMLJob of type Train
		"""
		job = self.job
		job.job_type = "train"
		return ActionableEventTypes.SCHEDULE_JOB.value, job
	
	def _done_everything(self, payload):
		"""
		Level 2 Callback to terminate session
		"""
		return ActionableEventTypes.TERMINATE.value, None
	
	def _average_iter(self, payload):
		"""
		Level 2 Callback to handle a DMLResult of an averaging job
		Should only return a DMLJob of type Train if averaging is done
		"""
		job = self.job
		self.listen_iterations += 1
		if self.listen_iterations > self.listen_bound:
			job.job_type = "train"
			return ActionableEventTypes.SCHEDULE_JOB.value, job
		else:
			return ActionableEventTypes.NOTHING.value, None

	def kickoff(self):
		"""
		Kickoff method for the FedAvgOptimizer. 
		"""
		return ActionableEventTypes.SCHEDULE_JOB.value, self.job

	def ask(self, event_type, payload):
		"""
		Parses an event dictionary into a level 1 callback.
		If the callback is not defined, it does nothing.
		Otherwise, calls the callback.
		"""
		callback = callback_handler(event_type, self.LEVEL_1_CALLBACKS)
		return callback(payload)

	def do_nothing(self, payload):
		pass