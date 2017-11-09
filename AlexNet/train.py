from mxnet.gluon import nn
import sys
sys.path.append('..')
import utils

from mxnet import init
from mxnet import gluon

# define network: AlexNet
net = nn.Sequential()
with net.name_scope():
    net.add(
        # first part
        nn.Conv2D(channels=96, kernel_size=11,
                  strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),

        # second part
        nn.Conv2D(channels=256, kernel_size=5,
                  padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),

        # third part
        nn.Conv2D(channels=384, kernel_size=3,
                  padding=1, activation='relu'),
        nn.Conv2D(channels=384, kernel_size=3,
                  padding=1, activation='relu'),
        nn.Conv2D(channels=256, kernel_size=3,
                  padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),

        # forth part. There is a dropout layer between two full connected layers
        nn.Flatten(),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),

        # fifth part. There is a dropout layer between two full connected layers
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),

        # six part
        nn.Dense(10)
    )

# get data. resize=224 means resize the input image from 28*28 to 224*224
train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=224)

# use gpu and use init.Xavier() to initialize the parameters of net.
ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

# define loss and optimizer
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

# train the model
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=5)