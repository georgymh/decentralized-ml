from transitions import Machine
from core.utils.event_types import ActionableEventTypes, CommMgrEventTypes
	
class FederatedAveragingOptimizer(Machine):
	# Define some states.
	# TODO: Fix states so that they compile; move 'after=' to Machine.add_transition() in init method
	def __init__(self, kwargs={}):
		'''
		Kwargs is a dictionary containing all the info needed to initialize the Optimizer.
		This includes any number of properties such as thresholds, etc.
		'''

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
			CommMgrEventTypes.SCHEDULE.name: self.handle_runner, 
			CommMgrEventTypes.NEW_WEIGHTS.name: self.handle_listen,
			CommMgrEventTypes.NOTHING.name: self.do_nothing,
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

	def increment_listen_iter(self):
		self.listen_iterations += 1
		if self.listen_iterations >= self.listen_bound:
			self.done_listening()

	def done_training(self):
		'''
		Optimizer is telling the CommMgr that it doesn't need to train anymore.
		Now, the CommMgr should communicate these results.
		'''
		self.total_iterations += 1
		if self.total_iterations >= self.total_bound:
			self.done_everything()

	def handle_runner(self):
		'''
		Doesn't do anything for now.
		'''
		self.train_iter()

	def handle_listen(self):
		'''
		Doesn't do anything for now.
		'''
		self.listen_iter()

	
	def kickoff(self):
		'''
		This is dumb, for now.
		'''
		return self.state

	def ask(self, payload):
		return self.parse(payload)

	def parse(self, event):
		"""
		Parses an event dictionary into a callback.
		If the callback is not defined, it does nothing.
		"""
		event_type = event.get('raw_event_type')
		callback = self.CALLBACKS[CommMgrEventTypes.NOTHING.name]
		if event_type in self.CALLBACKS:
			callback = self.CALLBACKS[event_type]
		callback(event)
		event['actionable_event_type'] = self.state
		return event

	def configure(self, CommMgr):
		# TODO: Determine whether this method is strictly necessary?
		self.CommMgr = CommMgr 
	def do_nothing(self, event):
		print("NOTHING")
'''
Everything below this line is nonsense but should not be commented out!
Ah...that's why it shouldn't be commented out.
'''

	# def schedule_job(self, job):
	# 	'''
	# 	Helper function to schedule a DMLJob via depdendency injection
	# 	'''
	# 	self.scheduler.add_job(job)
	# 	# TODO: Get the result of the job back to the Comm. Mgr. 

	# def make_job_from_event(self, event):
	# 	return deserialize_job(event)
		
	# def get_model_with_addr(self, model_addr):
	# 	'''
	# 	This Optimizer has been sent the address of a model that it should pick up
	# 	Use the IPFS method to deserialize the model (currently only Keras)
	# 	'''
	# 	# TODO: Be able to deserialize more than just Keras models
	# 	return ipfs2keras(self.neural_network, model_addr)
	
	# def on_enter_training(self, event):
	# 	model_addr = event.kwargs.get('model_addr')
	# 	self.neural_network = self.get_model_with_addr(model_addr)
	# 	print(self.model)
		
	# def on_exit_training(self, event):
	# 	print("result of training {}".format(self.neural_network))
		
	# def is_valid(self, event):
	# 	return True
	
	# def time_elapsed(self, event):
	# 	return self.time >= self.time_threshold
	
	# def is_also_valid(self):
	# 	return True



	# def do_listening(self, event):
	# 	'''
	# 	Event data should be whatever model the Optimizer has just received.
	# 	Optimizer is telling the CommMgr that it wants the CommMgr to remember this.
	# 	But also, to keep listening.
	# 	'''
	# def do_training(self, event):
	# 	'''
	# 	Event data should be the model to train on.
	# 	Optimizer is telling the CommMgr that it wants the CommMgr to train again.
	# 	'''
	# 	event['OptimizerEventType'] = OptimizerEventTypes.TRAIN.name
	# 	return event
	# 	# job = self.make_job_from_event(event)
	# 	# # TODO: Get the result of the job back to this Optimizer so that it knows what to do
	# 	# self.scheduler.add_job(job)

	# def do_validating(self, event):
	# 	'''
	# 	Event data should be the model to validate on.
	# 	Optimizer is telling the CommMgr that it wants the CommMgr to validate this model.
	# 	NOT USED IN MVP.
	# 	'''
	# 	event['OptimizerEventType'] = OptimizerEventTypes.VALIDATE.name
	# 	return event

	# def do_averaging(self, event):
	# 	'''
	# 	When there are many models to be averaged, the Optimizer needs to validate them,
	# 	and then do a weighted average.
	# 	This is the second step; the average.
	# 	'''
	# 	# Get the weights
	# 	event['OptimizerEventType'] = OptimizerEventTypes.AVERAGE.name
	# 	return event