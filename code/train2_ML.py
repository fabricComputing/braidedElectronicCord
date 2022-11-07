import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy
import pandas as pd
# from distributed.joblib import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, preprocessing
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import column_or_1d
import sklearn.metrics as sm
import joblib
import time

from sklearn import metrics
from sklearn.preprocessing import label_binarize

PRINT_confusion = 1
PRINT_report = 1
DRAW_confusion = 1

add5 = "./data_set/Wear-resistant/train/wear-resistant.csv"


def plot_aoc(ytest, pred_test_y_proba):
    fpr, tpr, auc = [], [], []
    y_one_hot = label_binarize(ytest, classes=[0, 1, 2, 3, 4, 5, 6])
    f2, t2, thresholds = metrics.roc_curve(
        y_one_hot.ravel(), pred_test_y_proba.ravel())  # 各个模型的真值和预测结果
    a = metrics.auc(f2, t2)

    fpr.append(f2)
    tpr.append(t2)
    auc.append(a)
    model_name = ['n_estimators = 200',
                  'n_estimators = 100', 'n_estimators = 10']
    color_name = ['b', '#c70039', '#32CD32', '#f08a5d', '#111d5e', 'b', 'g']
    ls_name = ['--', '-', '-.', '-.', '--', '-']
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
        classes = ["ca", "hd", "l", "m", "nd", "sj", "u", "zw"]
        # classes = ["ca&hd", "dj", "nd", "sj", "zw"]
    plt.rc('font', family='Tahoma', size='12')  # 设置字体样式、大小
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                                       'SimHei', 'Lucida Grande', 'Verdana']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.figure(figsize=(150, 150))
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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

    xlocations = np.array(range(len(classes)))
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.05)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

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

def normalize(lista):
    maxa = max(lista)
    mina = min(lista)
    result = []
    # print(len(lista))
    for i in lista:
        new = (i-mina)/(maxa - mina)
        result.append(new)
    return result

def test(address):
    x = pd.read_csv(address)
    y = x[['type']]
    # y[[y == 1]] = 0
    # y[[y == 3]] = 2
    # y[[y == 6]] = 2

    x = x.drop(['type'], axis=1)  # 删除标签是type的数据，以列为轴
    print(x.shape)
    x = x.fillna(method='ffill', axis=1)  # 用前一个值填充NAN，适用于dataframe数据
    x_new = []
    for item in range(len(x)):
        temp = normalize(np.array(x)[item])
        x_new.append(temp)
    x = x_new

    seed = 5
    test_size = 0.3
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=test_size, random_state=seed)

    xtrain = pd.DataFrame(xtrain)
    ytrain = column_or_1d(ytrain.values.ravel(), warn=True)
    ytest = column_or_1d(ytest.values.ravel(), warn=True)

    print("------------------------")

    # MLP多层感知机
    # time_a = time.time()
    clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(128, 64, 32), random_state=1)
    stratifiedkf = StratifiedKFold(n_splits=5)
    score=cross_val_score(clf, x,y,cv=stratifiedkf)
    # time_b = time.time()
    # print("time: ",time_b-time_a)
    print("cross validation scores are {}".format(score))
    print("average cross validation score :{}".format(score.mean()))
    clf = clf.fit(xtrain, ytrain)
    joblib.dump(clf, 'saved_model/clf.pkl')
    result = clf.score(xtrain, ytrain)
    print("MLP 的准确率", result)


    pred_test_y = clf.predict(xtest)


    m = sm.confusion_matrix(ytest, pred_test_y)
    if PRINT_confusion:
        print('混淆矩阵为：', m, sep='\n')
    r = sm.classification_report(ytest, pred_test_y)
    if PRINT_report:
        print('分类报告为：', r, sep='\n')
    if DRAW_confusion:
        plot_confusion_matrix(m, 'confusion_matrix_MLP.png',
                              title='MLP Confusion Matrix')

    # 随机森林分类
    rfc = RandomForestClassifier(n_estimators=1, criterion='entropy')  # 实例化
    stratifiedkf = StratifiedKFold(n_splits=5)
    score=cross_val_score(rfc, x,y,cv=stratifiedkf)
    print("cross validation scores are {}".format(score))
    print("average cross validation score :{}".format(score.mean()))
    rfc = rfc.fit(xtrain, ytrain)
    joblib.dump(rfc, 'saved_model/rfc.pkl')
    result = rfc.score(xtest, ytest)
    pred_test = rfc.predict(xtest)


    print("Random Forest Classifier 的准确率", result)
    pred_test_y = rfc.predict(xtest)  # 测试结果
    m = sm.confusion_matrix(ytest, pred_test_y)  # 打印混淆矩阵
    if PRINT_confusion:
        print('混淆矩阵为：', m, sep='\n')
    r = sm.classification_report(ytest, pred_test_y)
    if PRINT_report:
        print('分类报告为：', r, sep='\n')
    if DRAW_confusion:
        plot_confusion_matrix(
            m, 'confusion_matrix_RandomForest.png', title='Random Forest Confusion Matrix')

    # 逻辑回归算法
    LR = LogisticRegression(solver='sag', max_iter=20)
    stratifiedkf = StratifiedKFold(n_splits=5)
    score=cross_val_score(LR, x,y,cv=stratifiedkf)
    print("cross validation scores are {}".format(score))
    print("average cross validation score :{}".format(score.mean()))
    LR = LR.fit(xtrain, ytrain)
    joblib.dump(LR, 'saved_model/lr.pkl')

    result = LR.score(xtest, ytest)
    print("LogisticRegression", result)

    pred_test_y = LR.predict(xtest)  # 测试结果
    m = sm.confusion_matrix(ytest, pred_test_y)  # 打印混淆矩阵
    if PRINT_confusion:
        print('混淆矩阵为：', m, sep='\n')
    r = sm.classification_report(ytest, pred_test_y)
    if PRINT_report:
        print('分类报告为：', r, sep='\n')
    if DRAW_confusion:
        plot_confusion_matrix(m, 'confusion_matrix_LogisticRegression.png',
                              title='Logistic Regression Confusion Matrix')

    # SVM
    SVM = svm.SVC(C=1, kernel='rbf', gamma=0.5, decision_function_shape='ovr', probability=True)
    stratifiedkf = StratifiedKFold(n_splits=5)
    score=cross_val_score(SVM, x,y,cv=stratifiedkf)
    print("cross validation scores are {}".format(score))
    print("average cross validation score :{}".format(score.mean()))
    SVM = SVM.fit(xtrain, ytrain)
    joblib.dump(SVM, 'saved_model/svm.pkl')

    result = SVM.score(xtest, ytest)
    print("SVM 的准确率", result)

    pred_test_y = SVM.predict(xtest)  # 测试结果

    m = sm.confusion_matrix(ytest, pred_test_y)  # 打印混淆矩阵
    if PRINT_confusion:
        print('混淆矩阵为：', m, sep='\n')
    r = sm.classification_report(ytest, pred_test_y)
    if PRINT_report:
        print('分类报告为：', r, sep='\n')
    if DRAW_confusion:
        plot_confusion_matrix(m, 'confusion_matrix_SVM.png',
                              title='SVM Confusion Matrix')

    # K近邻算法
    KN = KNeighborsClassifier(n_neighbors=25)
    stratifiedkf = StratifiedKFold(n_splits=5)
    score=cross_val_score(KN, x,y,cv=stratifiedkf)
    print("cross validation scores are {}".format(score))
    print("average cross validation score :{}".format(score.mean()))
    KN = KN.fit(xtrain, ytrain)
    joblib.dump(KN, 'saved_model/kn.pkl')

    result = KN.score(xtest, ytest)
    print("K Neighbors Classifier 的准确率", result)

    pred_test_y = KN.predict(xtest)  # 测试结果

    m = sm.confusion_matrix(ytest, pred_test_y)  # 打印混淆矩阵
    if PRINT_confusion:
        print('混淆矩阵为：', m, sep='\n')
    r = sm.classification_report(ytest, pred_test_y)
    if PRINT_report:
        print('分类报告为：', r, sep='\n')
    if DRAW_confusion:
        plot_confusion_matrix(
            m, 'confusion_matrix_KNeighbors.png', title='K Neighbors Confusion Matrix')

    # 决策树
    DT = DecisionTreeClassifier()
    stratifiedkf = StratifiedKFold(n_splits=5)
    score=cross_val_score(DT, x,y,cv=stratifiedkf)
    print("cross validation scores are {}".format(score))
    print("average cross validation score :{}".format(score.mean()))
    DT = DT.fit(xtrain, ytrain)
    joblib.dump(DT, 'saved_model/dt.pkl')

    result = DT.score(xtest, ytest)

    # result = DT.score(xtest, ytest)
    print("Decision Tree Classifier的准确率", result)

    pred_test_y = DT.predict(xtest)  # 测试结果

    m = sm.confusion_matrix(ytest, pred_test_y)  # 打印混淆矩阵
    if PRINT_confusion:
        print('混淆矩阵为：', m, sep='\n')
    r = sm.classification_report(ytest, pred_test_y)
    if PRINT_report:
        print('分类报告为：', r, sep='\n')
    if DRAW_confusion:
        plot_confusion_matrix(
            m, 'confusion_matrix_DecisionTree.png', title='Decision Tree Confusion Matrix')

    print("-----------predict----------------")

class extract_festures():
    def __init__(self):
        self.signal = []
        self.energy = []
        self.entropy = []
        self.mean = []
        self.var = []
        self.standard_deviation = []
    def normalize(self,lista):
        maxa = max(lista)
        mina = min(lista)
        result = []
        for i in lista:
            new = (i-mina)/(maxa - mina)
            result.append(new)
        return result
    def signal2features(self,signal):
        nor_signal = self.normalize(signal)
        level = 3
        wp = pywt.WaveletPacket(data=nor_signal,wavelet = 'db1',mode='symmetric',maxlevel=level)
        for row in range(1, level+1):
          for i in [node.path for node in wp.get_level(row,'nature')]:
            self.signal.append(wp[i].data.tolist())
        self.calculate()
        print(self.energy, self.mean, self.var, self.entropy, self.standard_deviation)
    # def extract(self, folder_address, level):
    #     self.load_files(folder_address)
    #     for address in self.file_list:
    #         file_address = (folder_address + '/' + address)
    #         print(file_address)
    #         self.load_data(file_address, level)
    #         self.calculate()
    #         self.WP_signal = []
        # self.write_csv()


    def calculate(self):
        for i in range(len(self.signal)):
            item = self.signal[i]
            self.energy.append(pow(np.linalg.norm(item, ord=None), 2))
            self.mean.append(np.mean(item))
            self.var.append(np.var(item))
            self.entropy.append(self.Entropy(item))
            self.standard_deviation.append(np.std(item, ddof=1))
        # print(len(self.energy), len(self.mean))

    def Entropy(self, labels, base=2):
        # 计算概率分布
        probs = pd.Series(labels).value_counts() / len(labels)
        # 计算底数为base的熵
        en = stats.entropy(probs, base=base)
        return en

if __name__ == '__main__':
    address = add5
    test(address)

