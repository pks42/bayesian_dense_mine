# Drawn from https://gist.github.com/rocknrollnerd/c5af642cf217971d93f499e8f70fcb72 (in Theano)
# This is implemented in PyTorch
# Author : Anirudh Vemula

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

#from sklearn.datasets import fetch_mldata
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
import pandas as pd

def oversample(true, false):
    orig_true = np.empty_like(true)
    orig_true[:] = true
    mult = int(len(false) // len(true))
    remain = int(len(false) % len(true))
    for _ in range(mult-1):
        true = np.append(true, orig_true, axis = 0)
    if remain:
        true = np.append(true, orig_true[:remain], axis = 0)
    np.random.shuffle(true)
    return true, false

def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


class MLPLayer(nn.Module):
    def __init__(self, n_input, n_output, sigma_prior):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.sigma_prior = sigma_prior
        self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.W_logsigma = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.b_logsigma = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.lpw = 0
        self.lqw = 0

    def forward(self, X, infer=False, train_infer=False, sigma=0):
        if train_infer:
            sigma = torch.tensor([float(sigma)]).cuda()
            W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * (sigma*0.01)
            b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * (sigma*0.01)
            output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)
            return output

        if infer:
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_output)
            return output

        epsilon_W, epsilon_b = self.get_random()
        W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * epsilon_W
        b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * epsilon_b
        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)
        self.lpw = log_gaussian(W, 0, self.sigma_prior).sum() + log_gaussian(b, 0, self.sigma_prior).sum()
        self.lqw = log_gaussian_logsigma(W, self.W_mu, self.W_logsigma).sum() + log_gaussian_logsigma(b, self.b_mu, self.b_logsigma).sum()
        return output

    def get_random(self):
        return Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, self.sigma_prior).cuda()), Variable(torch.Tensor(self.n_output).normal_(0, self.sigma_prior).cuda())
        



class MLP(nn.Module):
    def __init__(self, n_input, n_output, sigma_prior):
        super().__init__()
        self.dropout_1 = nn.Dropout(0.8)
        self.dropout_2 = nn.Dropout(0.5)
        self.dropout_3 = nn.Dropout(0.2)
        self.l1 = MLPLayer(n_input, 200, sigma_prior)
        self.l1_relu = nn.ReLU()
        self.l2 = MLPLayer(200, 200, sigma_prior)
        self.l2_relu = nn.ReLU()
        self.l3 = MLPLayer(200, 200, sigma_prior)
        self.l3_relu = nn.ReLU()
        self.l4 = MLPLayer(200, n_output, sigma_prior)
        self.l4_sigmoid = nn.Sigmoid()
        # self.l1 = MLPLayer(n_input, 20, sigma_prior)
        # self.l1_relu = nn.ReLU()
        # self.l2 = MLPLayer(20, n_op, sigma_prior)
        # self.l2_sigmoid = nn.Sigmoid()

    def forward(self, X, infer=False, train_infer=False, sigma=0):
        if train_infer:
            output = self.l1_relu(self.l1(X, infer, train_infer, sigma))
            # output = self.l2_sigmoid(self.l2(X, infer, train_infer, sigma))
            output = self.l2_relu(self.l2(output, infer, train_infer, sigma))
            output = self.l3_relu(self.l3(output, infer, train_infer, sigma))
            output = self.l4_sigmoid(self.l4(output, infer, train_infer, sigma))
            return output
        
        output = self.l1_relu(self.l1(X, infer))
        output = self.dropout_1(output)
        # output = self.l2_sigmoid(self.l2(output, infer))
        output = self.l2_relu(self.l2(output, infer))
        output = self.dropout_2(output)
        output = self.l3_relu(self.l3(output, infer))
        output = self.dropout_3(output)
        output = self.l4_sigmoid(self.l4(output, infer))
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw + self.l4.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw + self.l4.lqw
        return lpw, lqw


def forward_pass_samples(X, y):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
    for _ in range(n_samples):
        output = net(X)
        sample_log_pw, sample_log_qw = net.get_lpw_lqw()
        sample_log_likelihood = log_gaussian(y, output, sigma_prior).sum()
        s_log_pw += sample_log_pw
        s_log_qw += sample_log_qw
        s_log_likelihood += sample_log_likelihood

    return s_log_pw/n_samples, s_log_qw/n_samples, s_log_likelihood/n_samples


def criterion(l_pw, l_qw, l_likelihood):
    return ((1./n_batches) * (l_qw - l_pw) - l_likelihood).sum() / float(batch_size)

def train_distrb_acc(network, x, bat_size, y_len, true_thres = 0.7):
    y_train = torch.randn(7,bat_size, y_len) 
    y_train[0] = network(x, infer=True).view(bat_size, y_len)
    sig = [-3,-2,-1,1,2,3]
    i = 1
    for j in sig:
        y_train[i] = network(x, train_infer=True, sigma=j).view(bat_size, y_len)
        i += 1
    thres = torch.ones(7, bat_size, y_len)
    thres = thres*true_thres
    pos_vec = torch.ge(y_train, thres).float()
    distrb_acc = pos_vec[0]*0.34 + pos_vec[1]*0.01325 + pos_vec[2]*0.07925 + pos_vec[3]*0.2375 + pos_vec[4]*0.2375 + pos_vec[5]*0.07925 + pos_vec[6]*0.01325
    return distrb_acc

#mnist = fetch_mldata('MNIST original')
combo_data = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_normalized_combo_nn_data.tsv', sep = '\t', encoding = 'utf-8')
like_check = False
#N = 5000
N = len(combo_data.iloc[:,0])


##STANDARD CHOOSE O/P COLUMNS
# data = np.float32(combo_data.iloc[:,[1,2,3,4,5,6,7]])
# idx = np.random.choice(data.shape[0], N)
# data = data[idx]
# target = np.int32(combo_data.iloc[:,8:18])
# target = target[idx]
# like_check = True
# data_like = np.float32(combo_data.iloc[:,18:28])
# data_like = data_like[idx]

##SINGLE O/P
like_check = True
drop_cols = ['salary','y_54','y_55','y_69','y_81','y_98','y_99','y_107','y_131','y_161','data_like_54','data_like_55','data_like_69','data_like_81','data_like_98','data_like_99','data_like_107','data_like_131','data_like_161']
combo_data_drop = combo_data.drop(columns = drop_cols)
true_data = combo_data_drop[combo_data_drop.y_157 == 1.0]
false_data = combo_data_drop[combo_data_drop.y_157 == 0.0]
# true_data, false_data = oversample(true_data, false_data)
all_data = np.append(true_data, false_data, axis = 0)
N_orig = len(all_data)
all_data[0]
data_orig = np.float32(all_data[:,:7])
target_orig = np.int32(all_data[:,7])
data_like = np.float32(all_data[:,8]).reshape(N_orig,1)
data_like_idx = np.where(data_like>0.0)[0]
true_check_data = data_orig[data_like_idx]
true_data_like = data_like[data_like_idx]

ada = ADASYN(random_state=42)
data, target = ada.fit_sample(data_orig, target_orig)
N = len(data)
target = target.reshape(N,1)


##FAKE DATA SET SIMPLE TEST
# # fake_data = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_single_op_FAKE_linear_data.tsv', sep ='\t', encoding='utf-8')
# fake_data = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Paper data\mijnspw_20160901-20170831_10171_single_op_FAKE_non_linear_data.tsv', sep ='\t', encoding='utf-8')
# data = np.float32(fake_data.iloc[:,0])
# N = data.shape[0]
# data = data.reshape(N,1)
# idx = np.random.choice(N, N)
# data = data[idx]
# target = np.int32(fake_data.iloc[:,-1])
# target = target[idx].reshape(N,1)

##SIMPLE TEST
# x_test = np.array([[5.90000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 3.47407e+03,
#                         6.66700e+01, 4.50240e+04, 1.10730e+04, 1.00000e+00],
#                     [3.90000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 4.57207e+03,
#                         1.00000e+02, 5.92550e+04, 4.51000e+02, 1.00000e+00],
#                     [6.40000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 2.98403e+03,
#                         7.50000e+01, 3.86740e+04, 2.50500e+03, 0.00000e+00],
#                     [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 2.55525e+03,
#                         8.88900e+01, 3.31170e+04, 1.59800e+04, 1.00000e+00],
#                     [5.60000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 4.62207e+03,
#                         1.00000e+02, 5.99030e+04, 1.57360e+04, 1.00000e+00],
#                     [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 3.02700e+03,
#                         1.00000e+02, 3.92300e+04, 1.09300e+03, 1.00000e+00],
#                     [5.20000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 4.22184e+03,
#                         8.88900e+01, 5.47160e+04, 1.38500e+04, 1.00000e+00],
#                     [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 2.55525e+03,
#                         8.88900e+01, 3.31170e+04, 1.59800e+04, 1.00000e+00],
#                     [6.40000e+01, 1.00000e+00, 1.00000e+00, 2.00000e+01, 3.82100e+03,
#                         8.88900e+01, 4.95210e+04, 2.17300e+03, 1.00000e+00],
#                     [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 4.14522e+03,
#                         1.00000e+02, 5.37230e+04, 2.74600e+03, 1.00000e+00],
#                     [6.40000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 2.98403e+03,
#                         7.50000e+01, 3.86740e+04, 2.50500e+03, 0.00000e+00],
#                     [6.20000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 2.89707e+03,
#                         1.00000e+02, 3.75470e+04, 8.16000e+02, 1.00000e+00],
#                     [6.00000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 7.96923e+03,
#                         1.00000e+02, 1.03282e+05, 1.89300e+03, 1.00000e+00],
#                     [6.50000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 8.39105e+03,
#                         1.00000e+02, 1.08749e+05, 4.14400e+03, 1.00000e+00],
#                     [5.40000e+01, 1.00000e+00, 3.00000e+00, 2.00000e+01, 6.26304e+03,
#                         1.00000e+02, 8.11690e+04, 1.76110e+04, 1.00000e+00],
#                     [4.70000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 3.75000e+03,
#                         1.00000e+02, 4.86000e+04, 1.56850e+04, 1.00000e+00]])
        
# y_test = np.array([0., 0.,  1., 1., 0.,  1., 0.,  1.,
#                     1.,  1.,  1., 0., 0.,  1., 0., 0.])

# N = len(x_test)
# data = x_test
# idx = np.random.choice(data.shape[0], N)
# data = data[idx]
# target = np.int32(y_test)
# target = target[idx].reshape(N, 1)


# train_idx, test_idx = train_test_split(np.array(range(N)), test_size=0.05)
# train_data, test_data = data[train_idx], data[test_idx]
# train_target, test_target = target[train_idx], target[test_idx]


#USING SECOND ANALYSIS AS TEST SET
test_pd = pd.read_csv(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Second analysis\spw_mijnspw_20180201-20180313_10171_normalized_combo_nn_data.tsv', sep = '\t', encoding = 'utf-8')
test_values = test_pd.drop(columns = ['salary','y_204', 'data_like_204']).values
test_data = np.float32(test_values[:,:7])
test_target = np.int32(test_values[:,7])
true_test = test_pd[test_pd.y_198 == 1.0]
train_data = data
train_target = target
idx = np.random.choice(train_data.shape[0], N)
train_data = train_data[idx]
train_target = train_target[idx]


n_input = train_data.shape[1]
n_op = train_target.shape[1]
M = train_data.shape[0]
sigma_prior = float(np.exp(-3))
# sigma_prior = float(1)
n_samples = 3
learning_rate = 0.001
n_epochs = 1000

#LOGGING RESULTS
file_name = 'test_18_y157_ADASYN_200_dropped'
file = open(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Troubleshooting\torched_tests\\' + file_name + '.txt', 'a+')

# Initialize network
net = MLP(n_input, n_op, sigma_prior)
net = net.cuda()
try:
    net.load_state_dict(torch.load(r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Troubleshooting\torched_tests\\' + file_name + '_model.pth'))
except FileNotFoundError:
    print("INITIALIZING NETWORK!")
# net = MLP(n_input, sigma_prior)

# building the objective
# remember, we're evaluating by samples
log_pw, log_qw, log_likelihood = 0., 0., 0.
batch_size = 100
n_batches = M / float(batch_size)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adadelta(net.parameters())

n_train_batches = int(train_data.shape[0] / float(batch_size))

for e in range(n_epochs):
    errs = []
    train_roc_auc = 0
    dist = 0
    for b in range(n_train_batches):
        net.zero_grad()
        X = Variable(torch.Tensor(train_data[b * batch_size: (b+1) * batch_size]).cuda())
        y = Variable(torch.Tensor(train_target[b * batch_size: (b+1) * batch_size]).cuda())
        
        log_pw, log_qw, log_likelihood = forward_pass_samples(X, y)
        loss = criterion(log_pw, log_qw, log_likelihood)
        errs.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()

    X = torch.Tensor(train_data).cuda()
    y = torch.Tensor(train_target).cuda()
    train_pred = net(X, infer=True)
    train_roc_auc = roc_auc_score(y.cpu().numpy(),train_pred.detach().cpu().numpy())

    if like_check:
            distrb_acc = train_distrb_acc(net, torch.Tensor(true_check_data).cuda(), len(true_check_data), n_op)
            data_like_bat = torch.Tensor(true_data_like)
            dist += torch.sum(torch.abs(data_like_bat-distrb_acc.cpu())).item()        


    X = Variable(torch.Tensor(test_data).cuda())
    with torch.no_grad():
        test_pred = net(X, infer=True)
    check, out = torch.max(test_pred, 1)
    # acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(test_target.ravel())) / float(test_data.shape[0])
    # print('epoch', e, 'loss', np.mean(errs), 'acc', acc)

    test_roc_auc = roc_auc_score(test_target,test_pred.detach().cpu().numpy())

    if like_check:
        op_msg = 'epoch->' + str(e)+ ' loss->' + str(np.mean(errs)) + ' train_roc_auc->' +str(train_roc_auc) + ' test_roc_auc->' + str(test_roc_auc) + ' likelihood_err->' + str(dist)
        file.write(op_msg + '\n')
        print(op_msg)
    else:
        op_msg = 'epoch->' + str(e)+ ' loss->' + str(np.mean(errs)) + ' train_roc_auc->' +str(train_roc_auc) + ' test_roc_auc->' + str(test_roc_auc)
        file.write(op_msg + '\n')
        print(op_msg)
    # if roc_auc > 0.59:
    #     print('CHECK VALUES!')
    #     break
file.close()

torch.save(net.state_dict(), r'C:\Users\GroeiFabriek\APG-Ateam\Data Personas\Troubleshooting\torched_tests\\' + file_name + '_model.pth')


# threshold = 0.9
# test_one_pred = torch.where(test_pred > threshold, torch.ones(test_pred.size()).cuda(), torch.zeros(test_pred.size()).cuda())
# confusion_matrix(test_target, test_one_pred.detach().cpu().numpy())

# train_roc_auc = roc_auc_score(y.cpu().numpy(),train_pred.detach().cpu().numpy())

# fpr, tpr, _ = roc_curve(y.cpu().numpy(),train_pred.detach().cpu().numpy())
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % train_roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

# threshold = 0.3
# train_one_pred = torch.where(train_pred > threshold, torch.ones(train_pred.size()).cuda(), torch.zeros(train_pred.size()).cuda())
# confusion_matrix(train_target, train_one_pred.detach().cpu().numpy())

# print(Variable(torch.Tensor(n_input, n_op).normal_(0, sigma_prior).cuda()))
# print(Variable(torch.Tensor(n_op).normal_(0, sigma_prior).cuda()))