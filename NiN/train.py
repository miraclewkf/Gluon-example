from mxnet.gluon import nn

import sys
sys.path.append('..')
import utils
from mxnet import gluon
from mxnet import init

def mlpconv(channels, kernel_size, padding,
            strides=1, max_pooling=True):
    out = nn.Sequential()
    out.add(
        nn.Conv2D(channels=channels, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1, activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1, activation='relu'))
    if max_pooling:
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    return out

net = nn.Sequential()
# add name_scope on the outer most Sequential
with net.name_scope():
    net.add(
        mlpconv(channels=96, kernel_size=11, padding=0, strides=4),
        mlpconv(channels=256, kernel_size=5, padding=2),
        mlpconv(channels=384, kernel_size=3, padding=1),
        nn.Dropout(.5),
        # 10 classes
        mlpconv(channels=10, kernel_size=3, padding=1, max_pooling=False),

        # use AvgPool2D layer to chanslate (batch_size,10,5,5) to (batch_size,10,1,1)
        nn.AvgPool2D(pool_size=5),

        # flatten to (batch_size,10)
        nn.Flatten()
    )

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=224)

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=1)