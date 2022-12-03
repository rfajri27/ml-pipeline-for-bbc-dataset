"""
Model hyperparameter tuner
"""

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs

from bbc_transform import (
    FEATURE_KEY,
    transformed_name
)

from bbc_trainer import (
    input_fn,
    model_builder,
    get_hyperparameters,
    vectorize_layer
)

NUM_EPOCHS = 5

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_categorical_accuracy",
    mode="max",
    verbose=1,
    patience=10,
)


def tuner_fn(fn_args: FnArgs):
    """Build the tuner using the KerasTuner API.
    Args:
        fn_args: Holds args as name/value pairs.
        - working_dir: working dir for tuning.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - schema_path: optional schema of the input data.
        - transform_graph_path: optional transform graph produced by TFT.
    Returns:
        A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                        model , e.g., the training and validation dataset. Required
                        args depend on the above tuner's implementation.
    """
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0],
                         tf_transform_output, NUM_EPOCHS)
    eval_set = input_fn(fn_args.eval_files[0],
                        tf_transform_output, NUM_EPOCHS)

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)]])

    tuner = kt.RandomSearch(
        model_builder,
        max_trials=6,
        hyperparameters=get_hyperparameters(),
        allow_new_entries=False,
        objective=kt.Objective('val_categorical_accuracy', 'max'),
        directory=fn_args.working_dir,
        project_name='bbc_classification')

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stopping_callback],
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
