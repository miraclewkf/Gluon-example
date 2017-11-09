from mxnet.gluon import nn
from mxnet import gluon
import sys
sys.path.append('..')
import utils

# define the network: LeNet. nn.Flatten() layer is used to flat the feature map for full connected layer,
# for example: flat (batch_size,50,2,2) to (batch_size,50*2*2)
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )

# initializaiton
ctx = utils.try_gpu()
net.initialize(ctx=ctx)
print('initialize weight on', ctx)

# load data
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# define loss and optimizer
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.5})

# train
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=5)