import logging
import random
import uuid
import time
import os
import pandas as pd

from core.configuration import ConfigurationManager
from custom.keras import model_from_serialized, get_optimizer
from data.iterators import count_datapoints
from data.iterators import create_random_train_dataset_iterator
from data.iterators import create_random_test_dataset_iterator
from core.utils.keras import train_keras_model, validate_keras_model
from core.utils.keras import serialize_weights
from core.utils.dmlresult import DMLResult
from core.utils.enums import JobTypes, callback_handler_no_default
# from core.utils.federated_learning_utils import federated_averaging
#       NOTE: Commented until the next PR is done.


logging.basicConfig(level=logging.DEBUG,
                    format='[Runner] %(asctime)s %(levelname)s %(message)s')

class DMLRunner(object):
    """
    DML Runner

    This class manages the training and validation of a DML job. It trains or
    validate a specified machine learning model on a local dataset. An instance
    of this class has a one-to-one relationship with a DML job, and gets freed up
    for another DML job after executing the job. Once free, it can schedule a new
    DML job from any DML session and repeat the cycle.

    Note that this class does not schedule how jobs received should be executed
    nor listens to incoming jobs. This class only executes the jobs.

    """

    def __init__(self, config_manager):
        """
        Sets up the unique identifier of the DML Runner and the local dataset path.
        """
        config = config_manager.get_config()
        self.iden = str(uuid.uuid4())[:8]
        
        self.config = dict(config.items("RUNNER"))
        self.JOB_CALLBACKS = {
            JobTypes.JOB_TRAIN.name: self._train,
            JobTypes.JOB_INIT.name: self._initialize_model,
            JobTypes.JOB_VAL.name: self._validate,
            JobTypes.JOB_TRANSFORM_SPLIT.name: self._transform_and_split_data
        }

    def run_job(self, job):
        """
        Identifies a DMLJob type and executes it.

        If the runner is already executing a job, it silently does nothing.
        """
        assert job.job_type.upper() in self.JOB_CALLBACKS, \
            'DMLJob type ({0}) is not valid'.format(job.job_type)
        logging.info("Running job (type: {0})...".format(job.job_type))
        callback = callback_handler_no_default(
            job.job_type,
            self.JOB_CALLBACKS,
        )
        try:
            results = callback(job)
        except Exception as e:
            logging.error("RunnerError: " + str(e))
            results = DMLResult(
                status='failed',
                job=job,
                error_message=str(e),
                results={},
            )
        self.current_job = None
        logging.info("Finished running job! (type: {0})".format(job.job_type))
        return results

    def _train(self, job):
        """
        Trains the specified machine learning model on all the local data,
        starting from the initial model state specified, until a stopping
        condition is met, and using the hyper-parameters specified.

        Returns a DMLResult with the updated model weights, the weighting factor
        omega, and stats about the training job.

        NOTE: Uses the same hyperparameters and labeler for training and
        validating during 'avg_type' of type 'val_acc'.

        NOTE2: Assumes 'job.weights' are the actual weights and not a path.
        """
        assert job.session_filepath, \
            "Transformed training and test sets not created yet!"
        train_dataset_path = os.path.join(
                                job.session_filepath, 
                                'train.csv'
                            )
        test_dataset_path = os.path.join(
                                job.session_filepath, 
                                'test.csv'
                            )
        data_count = count_datapoints(job.session_filepath)
        # Get the right dataset iterator based on the averaging type.
        avg_type = job.hyperparams['averaging_type']
        batch_size = job.hyperparams['batch_size']
        split = job.hyperparams['split']
        assert avg_type in ['data_size', 'val_acc'], \
            "Averaging type '{0}' is not supported.".format(avg_type)
        logging.info("Training model...")
        if avg_type == 'data_size':
            dataset_iterator = create_random_train_dataset_iterator(
                train_dataset_path,
                batch_size=batch_size,
                labeler=job.label_column_name,
            )
        elif avg_type == 'val_acc':
            dataset_iterator = create_random_train_dataset_iterator(
                train_dataset_path,
                batch_size=batch_size,
                labeler=job.label_column_name,
            )
            test_dataset_iterator = create_random_test_dataset_iterator(
                test_dataset_path,
                batch_size=batch_size,
                labeler=job.label_column_name,
            )

        # Train the model the right way based on the model type.
        assert job.framework_type in ['keras'], \
            "Model type '{0}' is not supported.".format(framework_type)
        if job.framework_type == 'keras':
            new_weights_path, train_stats = train_keras_model(
                job.serialized_model,
                job.weights,
                dataset_iterator,
                data_count*split,
                job.hyperparams,
                self.config
            )

        # Get the right omega based on the averaging type.
        if avg_type == 'data_size':
            omega = data_count * split
        elif avg_type == 'val_acc':
            val_stats = self._validate(
                job,
                custom_iterator=test_dataset_iterator
            ).results['val_stats']
            omega = val_stats['val_metric']['acc']
            train_stats.update(val_stats)

        # Return the results.
        # return new_weights_path, omega, train_stats
        results = DMLResult(
            status='successful',
            job=job,
            results={
                'weights': new_weights_path,
                'omega': omega,
                'train_stats': train_stats,
            },
            error_message="",
        )
        return results

    def _validate(self, job, custom_iterator=None):
        """
        Validates on all the local data the specified machine learning model at
        the state specified.

        Returns a DMLResult with the metrics returned by the model.

        NOTE: Assumes 'weights' are the actual weights and not a path.
        """
        logging.info("Validating model...")
        assert job.session_filepath, \
            "Transformed training and test sets not created yet!"
        train_dataset_path = os.path.join(
                                job.session_filepath, 
                                'train.csv'
                            )
        test_dataset_path = os.path.join(
                                job.session_filepath, 
                                'test.csv'
                            )
        data_count = count_datapoints(job.session_filepath)
        # Choose the dataset to validate on.
        batch_size = job.hyperparams['batch_size']
        split = job.hyperparams['split']
        if custom_iterator is None:
            dataset_iterator = create_random_test_dataset_iterator(
                test_dataset_path,
                batch_size=batch_size,
                labeler=job.label_column_name,
            )
        else:
            dataset_iterator = custom_iterator

        # Validate the model the right way based on the model type.
        assert job.framework_type in ['keras'], \
            "Model type '{0}' is not supported.".format(job.framework_type)
        if job.framework_type == 'keras':
            val_stats = validate_keras_model(
                job.serialized_model,
                job.weights,
                dataset_iterator,
                data_count*(1-split)
            )

        # Return the validation stats.
        results = DMLResult(
            status='successful',
            job=job,
            results={
                'val_stats': val_stats,
            },
            error_message="",
        )
        return results

    def _initialize_model(self, job):
        """
        Initializes and returns a DMLResult with the model
        weights as specified in the model.
        """
        assert job.framework_type in ['keras'], \
            "Model type '{0}' is not supported.".format(job.framework_type)
        logging.info("Initializing model...")
        if job.framework_type == 'keras':
            model = model_from_serialized(job.serialized_model)
            #model.summary()
            initial_weights = model.get_weights()
        results = DMLResult(
                    status='successful',
                    job=job,
                    results={
                        'weights': initial_weights,
                    },
                    error_message="",
                )
        return results


    def _transform_and_split_data(self, job):
        """
        Takes in a job, which should have the raw filepath assigned.

        1. In each folder, aggregate all data.
        2. Create session folder in transformed folder, along with data folders
           in session folder.
        3. Transform data in each raw data folder.
        4. Shuffle each transformed data and perform train-test split on each. 
        5. Put each training and test set in corresponding data folders in 
           session folder.
        6. Update session filepath in job.
        """

        # 1. Extracts all of the raw data from raw data filepath
        assert job.raw_filepath, \
            "Raw data filepath has not been set!"
        assert job.transform_function, \
            "Transform function has not been set!"
        files = os.listdir(job.raw_filepath)
        files = list(filter(lambda x: x.endswith('.csv'), files))
        assert len(files) == 1, \
            "Only supporting one file per dataset folder!"
        data_filepath = os.path.join(
                    job.raw_filepath,
                    files[0]
                )
        raw_data = pd.read_csv(data_filepath)

        # 2. Create transformed folder, if it doesn't exist.
        transformed_filepath = os.path.join(job.raw_filepath, "transformed")
        if not os.path.isdir(transformed_filepath):
            os.makedirs(transformed_filepath)

        # 3. Create session folder using timestamp and random characters.
        new_name = str(uuid.uuid4())
        session_filepath = os.path.join(transformed_filepath, new_name)
        os.makedirs(session_filepath)

        # 4. Retrieve transform function and train-test split from job.
        transform_function = job.transform_function
        split = job.hyperparams['split']

        # 5. Shuffle raw data, then split into train and test set.
        transformed_data = transform_function(raw_data)
        transformed_data = transformed_data.sample(frac=1)
        split_index = int(len(transformed_data)*split)
        train = transformed_data.iloc[:split_index] 
        test = transformed_data.iloc[split_index:]

        # 6. Create train.csv and test.csv in data folder.
        train.to_csv(
            os.path.join(session_filepath, 'train.csv'),
            index=False
        )

        test.to_csv(
            os.path.join(session_filepath, 'test.csv'),
            index=False
        )

        # 7. Return job with assigned session folder filepath
        results = DMLResult(
                    status='successful',
                    job=job,
                    results = {
                        'session_filepath': session_filepath
                    },
                    error_message=""
                )
        return results 

    def _get_raw_data(self, job):
        """
        Extracts all raw data from raw data filepath. Assumes filepath contains
        folders, which in turn contain csv files. 
        
        Returns dictionary of (key, value) pairs where key is a folder in the 
        directory and value is an aggregated DataFrame of the csv files in
        that folder. 
        """
        raw_dict = {}
        for folder in os.listdir(job.raw_filepath):
            folder_path = os.path.join(
                            os.path.abspath(job.raw_filepath), 
                            folder
                        )
            if not os.path.isdir(folder_path): continue
            files = os.listdir(folder_path)
            assert len(files) == 1, \
                "Only assuming one dataset per data folder!"
            filepath = os.path.join(folder_path, files[0])
            raw_dict[folder] = pd.read_csv(filepath)
        return raw_dict
