import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import column_or_1d
from itertools import product
from collections import OrderedDict, namedtuple
# jupyter中输出使用,可删除
from easydl import clear_output
from IPython.display import display
import time
import torch.utils.data as Data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# from d2l import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy
import pandas as pd
# from distributed.joblib import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, preprocessing
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from scipy import stats
import pywt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import column_or_1d
import sklearn.metrics as sm
import joblib

from sklearn import metrics
from sklearn.preprocessing import label_binarize

# rfc1 = joblib.load('saved_model/rfc.pkl')
np.random.seed(5)

PRINT_confusion = True
PRINT_report = True
DRAW_confusion = True
# CUDA_LAUNCH_BLOCKING=1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper-parameters
# sequence_length = 65
step = 2
input_size = 130/step
hidden_size = 128
num_layers = 4
num_classes = 5
batch_size = 65
num_epochs = 128
learning_rate = 0.001


# Recurrent neural network (many-to-one)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0.1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def normalize(lista):
    maxa = max(lista)
    mina = min(lista)
    result = []
    # print(len(lista))
    for i in lista:
        new = (i - mina) / (maxa - mina)
        result.append(new)
    return result


def plot_aoc(ytest, pred_test_y_proba):
    fpr, tpr, auc = [], [], []
    y_one_hot = label_binarize(ytest, classes=[0, 1, 2, 3, 4, 5, 6])
    f2, t2, thresholds = metrics.roc_curve(
        y_one_hot.ravel(), pred_test_y_proba.ravel())  # 各个模型的真值和预测结果
    a = metrics.auc(f2, t2)

    # 将所有模型跑出的结果都分别放在这三个列表中
    fpr.append(f2)
    tpr.append(t2)
    auc.append(a)
    model_name = ['n_estimators = 200',
                  'n_estimators = 100', 'n_estimators = 10']
    color_name = ['b', '#c70039', '#32CD32', '#f08a5d', '#111d5e', 'b', 'g']
    ls_name = ['--', '-', '-.', '-.', '--', '-']
    #
    # # 绘图
    # mpl.rcParams['font.sans-serif'] = u'SimHei'
    # mpl.rcParams['axes.unicode_minus'] = False
    for index, f in enumerate(fpr):
        t = tpr[index]
        a = auc[index]

        # FPR就是横坐标,TPR就是纵坐标
        plt.plot(f, t, c=color_name[index], ls=ls_name[index], lw=1.3, alpha=0.7,
                 label=model_name[index] + '(AUC=%.3f)' % a)
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.savefig('aoc.png')


def plot_confusion_matrix(cm, savename, classes=None, title='Normalized confusion matrix', cmap=plt.cm.Blues):
    if classes is None:
        # classes = ["ca", "hd", "l", "m", "nd", "sj", "u", "zw"]
        classes = ["ca&hd", "dj", "nd", "sj", "zw"]
    plt.rc('font', family='Tahoma', size='12')  # 设置字体样式、大小
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                                       'SimHei', 'Lucida Grande', 'Verdana']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.figure(figsize=(150, 150))
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=list(range(len(classes))), yticklabels=list(range(len(classes))),
           title=title,
           ylabel='Actual',
           xlabel='Predicted')
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.title(title, fontsize=17)

    # 通过绘制格网，模拟每个单元格的边框
    xlocations = np.array(range(len(classes)))
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.05)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) != 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white"
                        if cm[i, j] > thresh else "black", fontsize=12)
                # , fontsize=10)
    fig.tight_layout()
    plt.savefig(savename, format='png')
    plt.close(fig)


def test(address):
    x = pd.read_csv(address)
    y = x[['type']]
    y = np.array(y)
    # y[y == 1] = 0
    # y[y == 3] = 2
    # y[y == 6] = 2
    # y[y == 5] = 1
    # y[y == 7] = 3
    x = x.drop(['type'], axis=1)  # 删除标签是type的数据，以列为轴
    print(x.shape)
    x = x.fillna(method='ffill', axis=1)  # 用前一个值填充NAN，适用于dataframe数据
    x_new = []
    xnew = []
    ynew = []
    # print(x[0])
    for item in range(len(x)):
        temp = normalize(np.array(x)[item])
        xnew.append(temp)
        ynew.append(y[item])
    x = xnew
    y = ynew
    seed = 5
    test_size = 0.3
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=test_size, random_state=seed)

    xtrain = pd.DataFrame(xtrain)
    xtrain = xtrain.fillna(method='ffill', axis=1)

    ytrain = column_or_1d(ytrain, warn=True)
    ytest = column_or_1d(ytest, warn=True)

    # sampling_rate = 8 #每隔8帧采样一次数据

    torch_train_dataset = Data.TensorDataset(df_to_tensor1(xtrain), df_to_tensor2(ytrain))
    train_loader = Data.DataLoader(
        dataset=torch_train_dataset,
        batch_size=8,
        shuffle=True,
        # num_workers=2,
        drop_last=True,
    )

    torch_test_dataset = Data.TensorDataset(df_to_tensor1(xtest), df_to_tensor2(ytest))
    test_loader = Data.DataLoader(
        dataset=torch_test_dataset,
        batch_size=8,
        shuffle=True,
        # num_workers=2,
        drop_last=True,
    )

    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    total_step = len(train_loader)
    loss_all = []
    test_acc = []
    for epoch in range(num_epochs):
        for i, (signals, labels) in enumerate(train_loader):
            signals = signals.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels.flatten().long())
            print("loss", loss.detach().numpy())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        loss_all.append(int(100 * loss.item()))

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for signals, labels in test_loader:
                # images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                signals = signals.to(device)
                outputs = model(signals)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
        test_acc.append(int(100 * correct / total))


    print("loss_all is {}".format(loss_all))
    print("test_acc is {}".format(test_acc))



    # Test the model
    model_PATH = "LSTM_model.pt"

    torch.save(model, model_PATH)

    model = torch.load(model_PATH)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for signals, labels in test_loader:
            # images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            signals = signals.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format(100 * correct / total))

        # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    print("------------------------")


# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device

# convert a df to tensor to be used in pytorch
def df_to_tensor1(df):
    device = get_device()
    # print(type(df))
    a = np.array(df).shape
    new_np = np.array(df).reshape(a[0], 4, 130)[:, :, range(0, 130, step)]
    print('---------', new_np.shape)
    return torch.from_numpy(new_np).transpose(1, 2).swapaxes(1, 2).float().to(device)


def df_to_tensor2(df):
    device = get_device()
    return torch.from_numpy(np.array(df)).float().to(device)


if __name__ == '__main__':
    address = "" # train data address
    test(address)
