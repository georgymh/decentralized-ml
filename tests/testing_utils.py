from models.keras_perceptron    import KerasPerceptron
from custom.keras               import get_optimizer
from core.utils.enums           import JobTypes, RawEventTypes
from core.utils.dmljob          import DMLJob, serialize_job

def make_dataset_path():
    return 'datasets/mnist'

def make_model_json():
    m = KerasPerceptron(is_training=True)
    model_architecture = m.model.to_json()
    model_optimizer = get_optimizer(m.model)
    model_json = {
        "architecture": model_architecture,
        "optimizer": model_optimizer
    }
    return model_json

def make_initialize_job(model_json):
    initialize_job = DMLJob(
        JobTypes.JOB_INIT.name,
        model_json,
        "keras",
        hyperparams=make_hyperparams(),
        label_column_name='label'
    )
    return initialize_job

def make_transform_split_job( \
    model_json, raw_filepath, transform_function = lambda x: x):
    transform_split_job = DMLJob(
        JobTypes.JOB_TRANSFORM_SPLIT.name,
        model_json,
        "keras",
        hyperparams=make_hyperparams(),
        label_column_name='label',
        raw_filepath=raw_filepath,
        transform_function=transform_function
    )
    return transform_split_job

def make_hyperparams(split=0.004):
    hyperparams = {
        'averaging_type': 'data_size',
        'batch_size': 4,
        'epochs': 1,

        'split': split,
    }
    return hyperparams

def make_train_job(model_json, initial_weights, hyperparams, session_filepath):
    train_job = DMLJob(
        JobTypes.JOB_TRAIN.name,
        model_json,
        "keras",
        initial_weights,
        hyperparams,
        'label',
        session_filepath = session_filepath
    )
    return train_job

def make_serialized_job(session_filepath=None):
    initialize_job = make_initialize_job(make_model_json())
    initialize_job.session_filepath = session_filepath
    serialized_job = serialize_job(initialize_job)
    return serialized_job

def make_validate_job(model_json, new_weights, hyperparams, session_filepath):
    validate_job = DMLJob(
        JobTypes.JOB_VAL.name,
        model_json,
        'keras',
        new_weights,
        hyperparams,
        'label',
        session_filepath = session_filepath
    )
    return validate_job