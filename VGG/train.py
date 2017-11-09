from mxnet.gluon import nn
import sys
sys.path.append('..')
import utils
from mxnet import gluon
from mxnet import init

# define block
def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(
            nn.Conv2D(channels=channels, kernel_size=3,
                      padding=1, activation='relu')
        )
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

# define stack
def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out

num_outputs = 10
architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
net = nn.Sequential()
# add name_scope on the outermost Sequential
with net.name_scope():
    net.add(
        vgg_stack(architecture),
        nn.Flatten(),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        nn.Dense(num_outputs))

# load train_data
train_data, test_data = utils.load_data_fashion_mnist(batch_size=2, resize=224)

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

# define loss and optimizer
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.05})

# train
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=5)