"""Module for training and fitting an LSTM model for emotion estimation.

This module defines a manually constructed LSTM model with specific hyperparameters
derived from experimentation. It includes functionalities for compiling the model,
setting up callbacks for checkpoints and early stopping, and training the model
on provided training and validation datasets.
"""


import keras
import tensorflow as tf
from dotenv import load_dotenv
import os
import sys

# Add the 'src' directory to the Python path to resolve module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

from data.data_loading import load_training_data, load_validation
import datetime

# Configure TensorBoard logging for profiling and visualization
log_dir = "logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Profile batches 10 to 15 (inclusive) to collect performance data
tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    profile_batch='10, 15'
)

def train_model(x_train, y_train, x_val, y_val):
    """
    Trains the LSTM model with manually set hyperparameters.

    This function defines the LSTM model architecture, compiles it with a specific
    optimizer, loss function, and metrics. It also sets up model checkpoints
    and early stopping callbacks to optimize the training process. The model
    is then fitted using the provided training and validation data.

    Args:
        x_train (np.array): The blendshapes (features) of the training set.
        y_train (np.array): The one-hot encoded labels of the training set.
        x_val (np.array): The blendshapes (features) of the validation set.
        y_val (np.array): The one-hot encoded labels of the validation set.

    Returns:
        keras.Model: The trained Keras LSTM model.
    """
    # LSTM model architecture with hyperparameters determined through Keras Tuner experimentation
    model = keras.Sequential(
        [
            keras.layers.LSTM(
                units=34,
                activation="selu",
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.LSTM(
                units=26,
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=1.5774485705635927e-07),
            ),
            keras.layers.LSTM(
                units=14,
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=1.5774485705635927e-07),
            ),
            keras.layers.LSTM(
                units=14,
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=1.5774485705635927e-07),
            ),
            keras.layers.LSTM(
                units=14,
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=1.5774485705635927e-07),
            ),
            keras.layers.LSTM(
                units=14,
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=1.5774485705635927e-07),
            ),
            keras.layers.LSTM(
                units=30,
                activation="tanh",
                return_sequences=False,
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.Dense(units=3, activation="softmax"),
        ]
    )
    
    # Compile the model with AdamW optimizer and specified learning rate, weight decay, and metrics
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(), # Using categorical cross-entropy for one-hot encoded labels
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0007880621679373363,
            global_clipnorm=1,
            amsgrad=True,
            weight_decay=1.6709469947924033e-06,
        ),
        metrics=[
            keras.metrics.CategoricalCrossentropy(), # Monitor classification loss
            keras.metrics.CategoricalAccuracy(),     # Monitor classification accuracy
            keras.metrics.F1Score(average="macro"),  # Monitor F1-score with macro averaging
        ],
    )

    # Configure ModelCheckpoint callback to save the best model weights
    checkpoint_filepath = "ckpt/epoch_{epoch:02d}-val_loss_{val_loss:.4f}.keras"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",      # Monitor validation loss
            mode="min",          # Save model when validation loss is minimized
            save_best_only=True, # Only save the best model found so far
            verbose=0, # Suppress verbose output during checkpointing
        )

    # Configure EarlyStopping callback to halt training if no improvement is observed
    early_stop = keras.callbacks.EarlyStopping(
        monitor="categorical_crossentropy", # Monitor training categorical cross-entropy
        mode="min", # Stop when monitored quantity is minimized
        min_delta=0.0001, # Minimum change to be considered an improvement
        patience=500, # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored quantity
        verbose=0, # Suppress verbose output during early stopping
    )

    # Assigning weights for the classes (currently untuned and set to equal weights)
    weight_for_0 = 2.0   # happy
    weight_for_1 = 1.0   # unknown
    weight_for_2 = 2.0   # sad

    class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}

    # Fit the LSTM model to the training data
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=300, # Number of training epochs
        batch_size=150,
        verbose=2, # Verbosity mode (2 = one line per epoch)
        class_weight=class_weight, # Apply class weights during training
        callbacks=[model_checkpoint_callback, early_stop, tboard_callback], # Use defined callbacks
    )
    # Plot the final model structure and save it (optional, can be commented out)
    # keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    return model

if __name__ == "__main__":
    # Load training and validation data when the script is executed directly
    x_train, y_train = load_training_data()
    x_val, y_val = load_validation()
    # Train the model with the loaded data
    train_model(x_train, y_train, x_val, y_val)
