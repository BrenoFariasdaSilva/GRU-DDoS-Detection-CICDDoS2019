"""
================================================================================
GRU DDOS DETECTION CICDDOS2019 GRU MODEL AND TENSORFLOW DATASETS
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Builds the two-layer GRU network and batched tf.data datasets used by the GRU CICDDoS2019
    multi-class reproduction.

    Key features include:
        - Builds two 8-unit-configurable GRU layers with ReLU and sigmoid recurrent activation.
        - Applies configured dropout and 16/8-configurable dense hidden layers.
        - Compiles Softmax output with Adam and categorical cross-entropy.

Usage:
    1. Call build_gru() inside the selected TensorFlow device context.
    2. Build train/validation/test datasets with make_dataset().
    3. Train through gru_ddos_detection.experiment.

Outputs:
    - In-memory compiled Keras model and tf.data.Dataset objects.

TODOs:
    - None identified.

Dependencies:
    - tensorflow.
    - numpy.
    - gru_ddos_detection.config and tensorflow_runtime.

Assumptions & Notes:
    - The input shape remains one timestep by the published 20 selected features.
================================================================================
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .tensorflow_runtime import tf


def build_gru(cfg: Config, n_features: int, n_classes: int) -> tf.keras.Model:
    """
    Build and compile the supplied two-layer GRU classification model.

    :param cfg: Immutable experiment configuration containing model hyperparameters.
    :param n_features: Number of selected input features per one-timestep sample.
    :param n_classes: Number of Softmax output classes.
    :return: Compiled Keras GRU classification model.
    """

    inputs = tf.keras.Input(shape=(1, n_features), dtype=tf.float32, name="one_timestep_20_features")  # Preserve the original one-timestep recurrent input tensor
    recurrent = tf.keras.layers.GRU(
        cfg.gru_units,
        activation="relu",
        recurrent_activation="sigmoid",
        return_sequences=True,
        name="gru_1",
    )(inputs)  # Preserve first GRU configuration and sequence return behavior
    recurrent = tf.keras.layers.Dropout(cfg.dropout, name="dropout_1")(recurrent)  # Preserve dropout placement after the first GRU layer
    recurrent = tf.keras.layers.GRU(
        cfg.gru_units,
        activation="relu",
        recurrent_activation="sigmoid",
        return_sequences=False,
        name="gru_2",
    )(recurrent)  # Preserve second GRU configuration and final-sequence reduction
    recurrent = tf.keras.layers.Dropout(cfg.dropout, name="dropout_2")(recurrent)  # Preserve dropout placement after the second GRU layer
    hidden = tf.keras.layers.Dense(cfg.dense1, activation="relu", name="hidden_16")(recurrent)  # Preserve first dense hidden layer and original fixed layer name
    hidden = tf.keras.layers.Dense(cfg.dense2, activation="relu", name="hidden_8")(hidden)  # Preserve second dense hidden layer and original fixed layer name
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", dtype="float32", name="softmax_output")(hidden)  # Preserve float32 Softmax classification output
    model = tf.keras.Model(inputs, outputs, name="GRU_DDoS_Detection_CICDDoS2019")  # Preserve the original Keras model name
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )  # Preserve Adam, configured learning rate, categorical cross-entropy, and accuracy metric
    return model  # Return the compiled GRU model


def make_dataset(X: np.ndarray, y_onehot: np.ndarray, batch: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    """
    Build the bounded-prefetch TensorFlow dataset used for training or evaluation.

    :param X: Model-ready recurrent feature array.
    :param y_onehot: One-hot categorical label array aligned with X.
    :param batch: Batch size for dataset iteration.
    :param shuffle: Whether to shuffle with the original bounded buffer.
    :param seed: Random seed for TensorFlow dataset shuffling.
    :return: Batched tf.data.Dataset with one-batch prefetch.
    """

    dataset = tf.data.Dataset.from_tensor_slices((X, y_onehot))  # Create a tensor-slice dataset from aligned model inputs and categorical labels
    if shuffle:  # Verify if the dataset is intended for model fitting
        dataset = dataset.shuffle(min(len(X), 100_000), seed=seed, reshuffle_each_iteration=True)  # Preserve the original bounded shuffle buffer and reshuffling behavior
    return dataset.batch(batch, drop_remainder=False).prefetch(1)  # Preserve complete final batches and one-batch prefetch
