from mxnet.gluon import nn
import sys
sys.path.append('..')
import utils
from mxnet import autograd
from mxnet import gluon
from mxnet import nd

# define network
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Conv2D(channels=20, kernel_size=5))

    ### add BN layer. BN layer should be followed by activation layer
    net.add(nn.BatchNorm(axis=1))
    net.add(nn.Activation(activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))

    net.add(nn.Conv2D(channels=50, kernel_size=3))

    ### add BN layer. BN layer should be followed by activation layer
    net.add(nn.BatchNorm(axis=1))
    net.add(nn.Activation(activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))

    net.add(nn.Flatten())
    # full connect layer
    net.add(nn.Dense(128, activation="relu"))
    # full connect layer
    net.add(nn.Dense(10))

# use gpu
ctx = utils.try_gpu()
net.initialize(ctx=ctx)

# get data
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# define loss function and optimizer
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})


for epoch in range(5):
    # initial loss and accuracy
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            # forward
            output = net(data.as_in_context(ctx))
            # get loss
            loss = softmax_cross_entropy(output, label)
        # backword
        loss.backward()
        # Make one step of parameter update. The parameter batch_size means gradient will be normalized by 1/batch_size
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))