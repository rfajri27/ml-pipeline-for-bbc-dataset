"""
Builder & trainer modules to train the model
"""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
import os
import keras_tuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs
import absl

from bbc_transform import (
    FEATURE_KEY,
    LABEL_KEY,
    transformed_name,
)


def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern,
             tf_transform_output,
             num_epochs,
             batch_size=64) -> tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset


def get_hyperparameters() -> kt.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hp = kt.HyperParameters()
    # Defines search space.
    hp.Int("dense_units", min_value=32, max_value=256, step=32)
    hp.Int("dense_units_1", min_value=32, max_value=256, step=32)
    hp.Choice('learning_rate', [1e-2, 1e-3], default=1e-2)
    return hp


VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

EMBEDDING_DIM = 16

def model_builder(hp: kt.HyperParameters):
    """Build machine learning model"""

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY),
                            dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hp["dense_units"], activation='relu')(x)
    x = layers.Dense(hp["dense_units_1"], activation='relu')(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]

    )

    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Get serve tf example_fn"""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        # get predictions using the transformed features
        return model(transformed_features)
    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    """Train the model based on given args.
    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                          mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir, monitor='val_categorical_accuracy',
        mode='max', verbose=1, save_best_only=True)

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)]])

    if fn_args.hyperparameters:
        hp = kt.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        # This is a shown case when hyperparameters is decided and Tuner is removed
        # from the pipeline. User can also inline the hyperparameters directly in
        # _build_keras_model.
        hp = model_builder.get_hyperparameters()

    absl.logging.info('HyperParameters for training: %s' % hp.get_config())
    # Build the model
    model = model_builder(hp)

    # Train the model
    model.fit(x=train_set,
              validation_data=val_set,
              callbacks=[tensorboard_callback, es, mc],
              steps_per_epoch=fn_args.train_steps,
              validation_steps=fn_args.eval_steps,
              epochs=10)

    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        )
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)
