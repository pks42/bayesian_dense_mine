
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

#from sklearn.datasets import fetch_mldata
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import pandas as pd


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

    def forward(self, X, infer=False):
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

    def comp_infer_op(self, X, sigma, true_thres):
        sigma = torch.tensor([float(sigma)]).cuda()
        W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * sigma
        b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * sigma
        print(X.size())
        print(W.size())
        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)
        # output = b.expand(X.size()[0], self.n_output)
        # print(output)
        # if output >= true_thres:
        #     output = 1.0
        # else:
        #     output = 0.0
        return output

    def infer(self, X, true_thres = 0.7):
        # y_1 = self.comp_infer_op(X,-3, true_thres)
        # y_2 = self.comp_infer_op(X,-2, true_thres)
        # y_3 = self.comp_infer_op(X,-1, true_thres)
        # y_4 = self.comp_infer_op(X, 0, true_thres)
        # y_5 = self.comp_infer_op(X, 1, true_thres)
        # y_6 = self.comp_infer_op(X, 2, true_thres)
        y_7 = self.comp_infer_op(X, 2, true_thres)
        print(y_7)


sigma_prior = float(np.exp(-3))
net = MLPLayer(9, 200, sigma_prior)
net = net.cuda()

x_test = np.array([[5.90000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 3.47407e+03,
                        6.66700e+01, 4.50240e+04, 1.10730e+04, 1.00000e+00],
                    [3.90000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 4.57207e+03,
                        1.00000e+02, 5.92550e+04, 4.51000e+02, 1.00000e+00],
                    [6.40000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 2.98403e+03,
                        7.50000e+01, 3.86740e+04, 2.50500e+03, 0.00000e+00],
                    [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 2.55525e+03,
                        8.88900e+01, 3.31170e+04, 1.59800e+04, 1.00000e+00],
                    [5.60000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 4.62207e+03,
                        1.00000e+02, 5.99030e+04, 1.57360e+04, 1.00000e+00],
                    [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 3.02700e+03,
                        1.00000e+02, 3.92300e+04, 1.09300e+03, 1.00000e+00],
                    [5.20000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 4.22184e+03,
                        8.88900e+01, 5.47160e+04, 1.38500e+04, 1.00000e+00],
                    [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 2.55525e+03,
                        8.88900e+01, 3.31170e+04, 1.59800e+04, 1.00000e+00],
                    [6.40000e+01, 1.00000e+00, 1.00000e+00, 2.00000e+01, 3.82100e+03,
                        8.88900e+01, 4.95210e+04, 2.17300e+03, 1.00000e+00],
                    [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 4.14522e+03,
                        1.00000e+02, 5.37230e+04, 2.74600e+03, 1.00000e+00],
                    [6.40000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 2.98403e+03,
                        7.50000e+01, 3.86740e+04, 2.50500e+03, 0.00000e+00],
                    [6.20000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 2.89707e+03,
                        1.00000e+02, 3.75470e+04, 8.16000e+02, 1.00000e+00],
                    [6.00000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 7.96923e+03,
                        1.00000e+02, 1.03282e+05, 1.89300e+03, 1.00000e+00],
                    [6.50000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 8.39105e+03,
                        1.00000e+02, 1.08749e+05, 4.14400e+03, 1.00000e+00],
                    [5.40000e+01, 1.00000e+00, 3.00000e+00, 2.00000e+01, 6.26304e+03,
                        1.00000e+02, 8.11690e+04, 1.76110e+04, 1.00000e+00],
                    [4.70000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 3.75000e+03,
                        1.00000e+02, 4.86000e+04, 1.56850e+04, 1.00000e+00]])

X = Variable(torch.Tensor(x_test[0]).cuda())
X = X.view(1,9)

net.infer(X)