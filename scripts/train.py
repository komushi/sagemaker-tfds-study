import argparse
import os

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

tfds.disable_progress_bar()
tf.enable_v2_behavior()


EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TF_AUTOTUNE = tf.data.experimental.AUTOTUNE
TF_DATASET_NAME = 'mnist'
MODEL_VERSION = '1'

def _parse_args():
    parser = argparse.ArgumentParser()
    
    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)    
    parser.add_argument('--dataset_path', type=str)
    
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model_version', type=str, default=MODEL_VERSION)


    return parser.parse_known_args()

def create_model(learning_rate, num_classes):
    optimizer = Adam(learning_rate=learning_rate)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    model.summary()

    return model


def get_datasets(dataset_name, dataset_path):
    tfds.disable_progress_bar()

#     (ds_train, ds_test), ds_info = tfds.load(
#         dataset_name,
#         split=['train', 'test'],
#         shuffle_files=True,
#         as_supervised=True,
#         with_info=True
#     )    

    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name,
        split=['train', 'test'],
        data_dir=dataset_path,
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )    

    return (ds_train, ds_test), ds_info

def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255., label


def train_preprocess(ds_train, batch_size, num_examples):
    train_preprocessed = ds_train.map(normalize_img, num_parallel_calls=TF_AUTOTUNE)
    train_preprocessed = train_preprocessed.cache()
    train_preprocessed = train_preprocessed.shuffle(num_examples)
    train_preprocessed = train_preprocessed.batch(batch_size)
    train_preprocessed = train_preprocessed.prefetch(TF_AUTOTUNE)
    
    return train_preprocessed


def test_preprocess(ds_test, batch_size):
    test_preprocessed = ds_test.map(normalize_img, num_parallel_calls=TF_AUTOTUNE)
    test_preprocessed = test_preprocessed.batch(batch_size)
    test_preprocessed = test_preprocessed.cache()
    test_preprocessed = test_preprocessed.prefetch(TF_AUTOTUNE)    
    
    return test_preprocessed

if __name__ == "__main__":
    args, _ = _parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print(
        f"\nBatch Size = {batch_size}, Epochs = {epochs}, Learning Rate = {learning_rate}\n")

    print(
        f"\nmodel_dir = {args.model_dir}, \ndataset_path = {args.dataset_path}")

    
    (ds_train, ds_test), ds_info = get_datasets(TF_DATASET_NAME, args.dataset_path)
    NUM_EXAMPLES = ds_info.splits['train'].num_examples
    NUM_CLASSES = ds_info.features['label'].num_classes
    print(
        f"\nnum_classes = {NUM_CLASSES} \n")

    train_preprocessed = train_preprocess(ds_train, batch_size, NUM_EXAMPLES)
    test_preprocessed = test_preprocess(ds_test, batch_size)

    model = create_model(learning_rate, NUM_CLASSES)

    model.fit(train_preprocessed, 
              epochs=args.epochs,
              validation_data=test_preprocessed)    
    
    export_path = os.path.join(args.sm_model_dir, args.model_version)
    print(f"\nModel version: {args.model_version} exported to: {export_path}\n")

    model.save(export_path)