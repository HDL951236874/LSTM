# coding:utf-8
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch

data_set = np.loadtxt('sample_for_LSTM', dtype="float32")

# data_set_out = np.loadtxt('./data/1/outpostion/1.txt', dtype="float32")
'''choose the working mode'''
train = 0

retrain = 0

lookback = 4

input = 22

output = 2

train_percent = 1

epoch = 5000


def create_dataset(dataset, look_back=lookback):
    dataX, dataY = [], []
    temp_in, temp_out = [], []
    for i in range(len(dataset)):
        a = dataset[i][0:2]
        b = dataset[i][2:24]
        temp_in.append(b)
        temp_out.append(a)
    for i in range(len(dataset) - look_back):
        a = temp_in[i:(i + look_back)]
        b = temp_out[i + look_back - 1]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


data_X, data_Y = create_dataset(data_set)

# split data set
train_size = int(len(data_X) * train_percent)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

train_X = train_X.reshape(-1, lookback, input)
train_Y = train_Y.reshape(-1, 1, output)
train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)


class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.line1 = nn.Linear(hidden_dim, hidden_dim)
        self.line2 = nn.Linear(hidden_dim, n_class)
        if retrain:
            re_model_stic = torch.load("./lstm_stic.pth")
            self.load_state_dict(re_model_stic)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.line1(out)
        out = self.line2(out)
        return out


if train:
    model = Rnn(input, 64, 3, output)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    loss = None
    total_epoch = epoch
    for epoch in range(total_epoch):
        if torch.cuda.is_available():
            train_x = train_x.cuda()
            train_y = train_y.cuda()
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        out = model(var_x)
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print('epoch {}, loss is {}'.format(epoch + 1, loss.data[0]))
        loss = loss.data[0]

    torch.save(model.state_dict(), './lstm_stic.pth')
    print "Saved The Model"
else:
    test_model = Rnn(input, 64, 3, output)
    # test_model = test_model.cuda()
    test_model_stic = torch.load("./lstm_stic.pth")
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in test_model_stic.items():
        name = k
        new_state_dict[name] = v
    test_model.load_state_dict(test_model_stic)
    data_X = data_X.reshape(-1, lookback, input)
    data_X = torch.from_numpy(data_X)
    # var_data = Variable(data_X).cuda()
    var_data = Variable(data_X)
    predict = test_model(var_data)

    predict = predict.cpu().data.numpy()

    predict = predict.reshape(-1, output)

    lable_1 = data_Y[:, 0]
    lable_2 = data_Y[:, 1]

    X = np.arange(0, len(lable_1), 1)

    out_1 = predict[:, 0]
    out_2 = predict[:, 1]
    plt.figure(1)

    plt.subplot(211)
    plt.plot(lable_1, 'b')
    plt.plot(out_1, 'r')
    plt.subplot(212)
    plt.plot(lable_2, 'b')
    plt.plot(out_2, 'r')

    plt.figure(2)
    plt.subplot(211)
    plt.scatter(X, lable_1, c='b')
    plt.scatter(X, out_1, c='r')
    plt.grid(True)
    plt.subplot(212)
    plt.scatter(X, lable_2, c='b')
    plt.scatter(X, out_2, c='r')
    plt.grid(True)
    plt.grid(True)

    plt.figure(3)
    plt.subplot(211)
    plt.plot(out_1 - lable_1, 'r')
    plt.subplot(212)
    plt.plot(out_2 - lable_2, 'r')
    plt.show()
