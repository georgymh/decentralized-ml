from transitions import Machine
from core.utils.event_types import ActionableEventTypes, RawEventTypes
	
class FederatedAveragingOptimizer(Machine):
	"""
	Federated Averaging Optimizer

	Implements the Federated Averaging algorithm (with some tweaks)
	and acts as a "brain" for the Unix Service
	"""
	def __init__(self, content):
		'''
		Kwargs is a dictionary containing all the info needed to initialize the Optimizer.
		This includes any number of properties such as thresholds, etc.
		'''
		kwargs = content.get('optimizer_params')
		serialized_job = content.get('serialized_job')
		# The Optimizer knows its job
		self.serialized_job = serialized_job
		# What have we accomplished today?
		self.training_iterations = kwargs.get('training_iterations', 0)
		self.listen_iterations = kwargs.get('listen_iterations', 0)
		self.total_iterations = kwargs.get('total_iterations', 0)
		self.train_bound = kwargs.get('train_bound', 1)
		self.listen_bound = kwargs.get('listen_bound', 1)
		self.total_bound = kwargs.get('total_bound', 2)
		self.training_history = []
		states = kwargs.get('states',
				['splitting', 'training', 'communicating', 'averaging', 'terminate'])
		transitions = kwargs.get('transitions',
			[
			# {'train_iter', 'training', None, after='increment_train_iter'},
			# ['failure', ['splitting', 'training', 'communicating', 'averaging'], '=', after='raise_exception'],
			# ['train_iter', 'training', None, after='increment_train_iter'],
			['done_training', 'training', 'communicating'],
			# ['listen_iter', '*', None, after='increment_listen_iter'],
			['done_listening', 'communicating', 'averaging'],
			['done_averaging', 'averaging', 'training'],
			['done_splitting', 'splitting', 'training'],
			['done_everything', '*', 'terminate']])
		# Initialize the state machine
		Machine.__init__(self, 
							   states=states, 
							   transitions=transitions, 
							   initial='splitting')
		self.add_transition('train_iter', 'training', None, after='increment_train_iter')
		self.add_transition('listen_iter', '*', None, after='increment_listen_iter')
		self.CALLBACKS = {
			RawEventTypes.SPLIT.value: self.done_splitting,
			RawEventTypes.JOB_FINISHED.value: self.handle_runner, 
			RawEventTypes.NEW_WEIGHTS.value: self.handle_listen,
			RawEventTypes.NOTHING.value: self.do_nothing,
			RawEventTypes.TERMINATE.value: self.done_everything
		}
		
		
	def increment_train_iter(self):
		'''
		Tells the Optimizer that it's trained.
		Most of the time this should be enough, because the Optimizer will
		give one DMLJob (with sgd_iter=50 or so).
		But sometimes it won't.
		'''
		# self.training_history.append(delta)
		self.training_iterations += 1
		if self.training_iterations >= self.train_bound:
			self.done_training()
			return self.state
		else:
			return "NOTHING"

	def increment_listen_iter(self):
		self.listen_iterations += 1
		if self.listen_iterations >= self.listen_bound:
			self.done_listening()
			return self.state
		else:
			return "NOTHING"

	# def done_training(self):
	# 	'''
	# 	Optimizer is telling the CommMgr that it doesn't need to train anymore.
	# 	Now, the CommMgr should communicate these results.
	# 	'''
	# 	self.total_iterations += 1
	# 	if self.total_iterations >= self.total_bound:
	# 		self.done_everything()
	# 		return self.state
	# 	else:


	def handle_runner(self, args=None):
		'''
		Doesn't do anything for now.
		'''
		self.train_iter()

	def handle_listen(self, args=None):
		'''
		Doesn't do anything for now.
		'''
		self.listen_iter()
	
	def kickoff(self):
		'''
		This is dumb, for now.
		'''
		actionable_event = {}
		actionable_event['actionable_event_type'] = self.state
		return actionable_event

	def ask(self, payload):
		return self.parse(payload)

	def parse(self, event):
		"""
		Parses an event dictionary into a callback.
		If the callback is not defined, it does nothing.
		"""
		event_type = event.get('raw_event_type')
		# print(event_type)
		callback = self.CALLBACKS[RawEventTypes.NOTHING.name]
		if event_type in self.CALLBACKS:
			callback = self.CALLBACKS[event_type]
		callback(event)
		event['actionable_event_type'] = self.state
		event['job'] = self.serialized_job
		return event

	def do_nothing(self, event):
		print("NOTHING")