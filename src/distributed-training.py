import tensorflow as tf
import numpy as np
import os
import json

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the images

# Parse the TF_CONFIG environment variable
tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
num_workers = len(tf_config.get('cluster', {}).get('worker', []))

# Set up the distribution strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Define the model building function
def build_and_compile_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the per-worker batch size
per_worker_batch_size = 64
global_batch_size = per_worker_batch_size * num_workers

# Wrap the model creation in the strategy's scope
with strategy.scope():
    multi_worker_model = build_and_compile_model()

# Define the checkpoint directory
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Define callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True)
]

# Train the model
multi_worker_model.fit(x_train, y_train,
                       epochs=10,
                       batch_size=global_batch_size,
                       callbacks=callbacks,
                       validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = multi_worker_model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Save the model
if strategy.cluster_resolver.task_type == 'chief':
    multi_worker_model.save('./saved_model')
    print("Model saved")