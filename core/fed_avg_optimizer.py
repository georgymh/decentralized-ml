import logging

from core.utils.event_types import ActionableEventTypes, RawEventTypes
from core.utils.event_types import JobTypes, callback_handler
from core.utils.dmljob import deserialize_job
from core.utils.keras import serialize_weights
from core.utils.dmlresult import DMLResult


logging.basicConfig(level=logging.DEBUG,
	format='[FedAvgOpt] %(asctime)s %(levelname)s %(message)s')


class FederatedAveragingOptimizer(object):
	"""
	Federated Averaging Optimizer

	Implements the Federated Averaging algorithm (with some tweaks)
	and acts as a "brain" for the Unix Service

	Specifically, upon hearing a new gradient, it will schedule
	a running average and increment listen_iterations until it exceeds
	listen_bound, at which point it will start retraining

	Breaks down info into Level_1 and Level_2 callbacks. 
	Level_1 Callbacks determine which Level_2 Callback to call,
	and do some Level_2_ -specific action to make the payload
	ready for the Level_2 Callback.

	"""
	def __init__(self, content):
		"""
		Kwargs is a dictionary containing all the info needed to initialize the Optimizer.
		This includes any number of properties such as thresholds, etc.
		"""
		logging.info("Setting up Optimizer")
		kwargs = content.get('optimizer_params')
		serialized_job = content.get('serialized_job')
		self.job = deserialize_job(serialized_job) # The Optimizer knows its job
		self.listen_iterations = kwargs.get('listen_iterations', 0)
		self.listen_bound = kwargs.get('listen_bound', 1)
		self.LEVEL_1_CALLBACKS = {
			RawEventTypes.JOB_DONE.name: self._handle_job_done,
			RawEventTypes.NEW_INFO.name: self._handle_new_info,
			RawEventTypes.NOTHING.name: self._do_nothing,
		}
		self.LEVEL_2_JOB_CALLBACKS = {
			JobTypes.JOB_TRAIN.name: self._done_training,
			JobTypes.JOB_INIT.name: self._done_initializing,
			JobTypes.JOB_AVG.name: self._average_increment,
			JobTypes.NOTHING.name: self._do_nothing,
		}
		self.LEVEL_2_INFO_CALLBACKS = {
			RawEventTypes.NEW_WEIGHTS.name: self._handle_new_weights,
			RawEventTypes.TERMINATE.name: self._terminate_session,
			RawEventTypes.NOTHING.name: self._do_nothing,
		}
		logging.info("Optimizer has been set up!")
	
	def kickoff(self):
		"""
		Kickoff method for the FedAvgOptimizer.
		Should split a dataset, and initialize the model.
		"""
		self.job.job_type = JobTypes.JOB_INIT.name
		return ActionableEventTypes.SCHEDULE_JOB.name, self.job

	def ask(self, event_type, payload):
		"""
		Parses an event dictionary into a level 1 callback.
		If the callback is not defined, it does nothing.
		Otherwise, calls the callback.
		"""
		callback = callback_handler(event_type, self.LEVEL_1_CALLBACKS)
		return callback(payload)
	
	def _handle_job_done(self, result):
		"""
		Level 1 Callback to handle the payload of a DMLResult
		Parses job.job_type into Level 2 Callback
		Also updates the weights
		"""
		assert isinstance(result, DMLResult)
		# assume success for now
		new_weights = result.results.get('weights')
		self._update_weights(new_weights)
		job_type = result.job.job_type
		callback = callback_handler(job_type, self.LEVEL_2_JOB_CALLBACKS)
		return callback(result)

	def _handle_new_info(self, payload):
		"""
		Level 1 Callback to handle the info from the network
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
		"""
		self.job.weights = weights

	def _handle_new_weights(self, payload):
		"""
		Level 2 Callback to handle new weights info
		Updates state and then returns appropriate DMLJob
		# TODO: Will be updated with Averaging PR
		"""
		self.job.job_type = JobTypes.JOB_AVG.name
		return ActionableEventTypes.SCHEDULE_JOB.name, self.job
	
	def _done_training(self, payload):
		"""
		Level 2 Callback to handle a DMLResult of a training job
		Should return a DMLJob of type Communication
		"""
		self.job.job_type = JobTypes.JOB_COMM.name
		return ActionableEventTypes.SCHEDULE_JOB.name, self.job
	
	def _done_initializing(self, payload):
		"""
		Level 2 Callback to handle a DMLResult of an initialization job
		Should return a DMLJob of type Train
		"""
		self.job.job_type = JobTypes.JOB_TRAIN.name
		return ActionableEventTypes.SCHEDULE_JOB.name, self.job
	
	def _terminate_session(self, payload):
		"""
		Level 2 Callback to terminate session
		"""
		return ActionableEventTypes.TERMINATE.name, None
	
	def _average_increment(self, payload):
		"""
		Level 2 Callback to handle a DMLResult of an averaging job
		Should only return a DMLJob of type Train if averaging is done
		Currently, we'll only support one round of fed. learning
		Will change in next PR
		"""
		self.listen_iterations += 1
		if self.listen_iterations > self.listen_bound:
			# self.listen_iterations = 0
			self.job.job_type = JobTypes.JOB_TRAIN.name
			return ActionableEventTypes.SCHEDULE_JOB.name, self.job
		else:
			return ActionableEventTypes.NOTHING.name, None

	def _do_nothing(self, payload):
		return None, None