import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

from clearml import Task

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.summary()
    return model


def train(model, ds_train, ds_test, epochs):
    model.fit(ds_train, epochs=epochs, validation_data=ds_test)
    return model


def infer(model, x):
    return model.predict(x)


def save(model, path):
    model.save(path, include_optimizer=False)


def get_info(model):
    model.input
    model.output


def get_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    return ds_train, ds_test, ds_info


def preprocess(image):
    img = Image.open('/content/7.png').convert('L')
    img = img.resize((28, 28))
    imgArr = np.asarray(img) / 255
    imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
    imgArr = imgArr.astype(np.float32)
    return imgArr


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def create_config_pbtxt(model, config_pbtxt_file, clearml_task):
    # name = clearml_task + ' - mnist_model'
    platform = "tensorflow_savedmodel"
    input_name = model.input_names[0]
    output_name = model.output_names[0]
    input_data_type = "TYPE_FP32"
    output_data_type = "TYPE_FP32"
    input_dims = str(model.input.shape.as_list()).replace("None", "-1")
    output_dims = str(model.output.shape.as_list()).replace("None", "-1")

    config_pbtxt = """
        platform: "%s"
        input [
            {
                name: "%s"
                data_type: %s
                dims: %s
            }
        ]
        output [
            {
                name: "%s"
                data_type: %s
                dims: %s
            }
        ]
    """ % (
        # name,
        platform,
        input_name, input_data_type, input_dims,
        output_name, output_data_type, output_dims
    )

    with open(config_pbtxt_file, "w") as config_file:
        config_file.write(config_pbtxt)



def main():
    parser = argparse.ArgumentParser(description='MNIST example')
    parser.add_argument('--clearml-project', type=str, default='mnist_testing1', help='project name')
    parser.add_argument('--clearml-task', type=str, default='mnist_testing_task1', help='task name')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 6)')
    args = parser.parse_args()

    # create clearml task
    task = Task.init(project_name=args.clearml_project, task_name=args.clearml_task, output_uri=True)

    # get data
    ds_train, ds_test, ds_info = get_data()

    batch_size=args.batch_size
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = get_model()
    model = train(model, ds_train, ds_test, epochs=args.epochs)
    save(model, "mnist_model")
    print("Model saved...!!!")

    # create the config.pbtxt for triton to be able to serve the model
    create_config_pbtxt(model=model, config_pbtxt_file='config.pbtxt', clearml_task=args.clearml_task)
    # store the configuration on the creating Task,
    # this will allow us to skip over manually setting the config.pbtxt for `clearml-serving`
    task.connect_configuration(configuration=Path('config.pbtxt'), name='config.pbtxt')


if __name__ == '__main__':
    main()
