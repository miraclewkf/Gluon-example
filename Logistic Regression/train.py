import sys
sys.path.append('..')
import utils

# load data
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

from mxnet import gluon

# define the network
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))
net.initialize()

# define the loss
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# define the optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

from mxnet import ndarray as nd
from mxnet import autograd

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    # data and label are NDArray
    for data, label in train_data:
        with autograd.record():
            # forward
            output = net(data)
            # calculate loss
            loss = softmax_cross_entropy(output, label)
        # backword
        loss.backward()
        # Make one step of parameter update. The parameter batch_size means gradient will be normalized by 1/batch_size
        trainer.step(batch_size)

        # get mean loss and return a scalar whose value is copied from this array
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))