import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import plotly.express as px
import json

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001, **kwargs):
        super(OrthogonalRegularizer, self).__init__(**kwargs)
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        config = {"num_features":self.num_features,
                  "l2reg":self.l2reg}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def get_class_map(data_path="data/ModelNet10/class_map.json"):
    class_map = json.load(open(data_path))
    class_map = {int(k):v for k,v in class_map.items()}
    return class_map

def get_pretrained_model(addr="model/pointnet_modelnet10"):
    return keras.models.load_model(addr, custom_objects={'OrthogonalRegularizer':OrthogonalRegularizer})

def get_dataset(train_addr='data/ModelNet10/train_data', test_addr='data/ModelNet10/test_data'):
    train_dataset = tf.data.experimental.load(train_addr)
    test_dataset = tf.data.experimental.load(test_addr)
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset()
    pretrained_model = get_pretrained_model()
    class_map = get_class_map()
    for points, labels in test_dataset:
        preds = tf.math.argmax(pretrained_model.predict(points), -1)
        print(preds)
        print(labels)
        print([class_map[idx] for idx in preds.numpy()])
        print([class_map[idx] for idx in labels.numpy()])
        break
    # import pdb;pdb.set_trace()

    # DATA_DIR = "/home/airbagy/.keras/datasets/ModelNet10"
    # NUM_POINTS = 2048
    # NUM_CLASSES = 10
    # BATCH_SIZE = 32
    # CLASS_MAP = get_class_map(NUM_POINTS)
    # model = keras.models.load_model('model/pointnet_modelnet10', custom_objects={'OrthogonalRegularizer':OrthogonalRegularizer})
    #
    # points, labels = list(test_dataset.take(1))[0]
    # model.compile(
    #     loss="sparse_categorical_crossentropy",
    #     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #     metrics=["sparse_categorical_accuracy"],
    # )
    #
    # model.fit(train_dataset, epochs=30, validation_data=test_dataset)


    # data = test_dataset.take(1)
    #
    # points, labels = list(data)[0]
    # points = points[:8, ...]
    # labels = labels[:8, ...]
    #
    # # run test data through model
    # preds = model.predict(points)
    # preds = tf.math.argmax(preds, -1)
    #
    # points = points.numpy()
    #
    # # plot points with predicted class and label
    # fig = plt.figure(figsize=(15, 10))
    # for i in range(8):
    #     ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    #     ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    #     ax.set_title(
    #         "pred: {:}, label: {:}".format(
    #             CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
    #         )
    #     )
    #     ax.set_axis_off()
    # plt.show()
