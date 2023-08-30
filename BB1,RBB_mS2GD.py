from libsvm.python.svmutil import *
import numpy as np
import pickle
from sklearn.preprocessing import normalize
import random
import matplotlib.pyplot as plt
import seaborn as sns
def data_generate(n, d, input_file="", output_file=" ",):
    # n: sample sizes
    # d: data dimensions
    y_train, x_train = svm_read_problem(input_file)
    temp = []
    for row in x_train:
        for i in range(1, d+1):
            row.setdefault(i, 0)
        temp.append([row[i] for i in range(1, d+1)])
    x_train = np.array(temp, dtype='float32').reshape(n, d)
    y_train = np.array(y_train, dtype='float32')
    f = open(output_file, 'wb')
    pickle.dump([x_train, y_train], f)
    f.close()
# data_generate(8124, 112, input_file=r"D:\libsvmdataset\mushrooms.txt", output_file=r"D:\dataset\mushrooms.txt")

class DataHolder:
    def __init__(self, dataset, lam1=0., lam2=0.):
        self.num = None
        self.dim = None
        self.train_set = None
        self.lam1 = lam1
        self.lam2 = lam2
        self.load_dataset(dataset)

    def load_dataset(self, dataset):
        if dataset == 'phishing':
            with open(r"D:\dataset\phishing.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'covtype':
            with open(r"D:\dataset\covtype.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'a8a':
            with open(r"D:\dataset\a8a.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'w8a':
            with open(r"D:\dataset\w8a.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'ijcnn1':
            with open(r"D:\dataset\ijcnn1.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'w7a':
            with open(r"D:\dataset\w7a.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'mushrooms':
            with open(r"D:\dataset\mushrooms.txt", 'rb') as g:
                train_set = pickle.load(g)
        self.train_set = [normalize(train_set[0], axis=1, norm='l2'), train_set[1]]
        self.dim = self.train_set[0].shape[1]
        self.num = self.train_set[0].shape[0]

    def logistic_indiv_function(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        c = np.sum(np.log(np.exp(-data.dot(w) * target) + 1)) / len(batch)
        return c

    def logistic_indiv_grad(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        c = (np.exp(-data.dot(w) * target) * (-target)) / (np.exp(-data.dot(w) * target) + 1)
        g = np.sum(np.einsum("nm,n->nm", data, c), axis=0) / len(batch)
        return g + self.lam2 * w


lam = 10**(-2)
holder = DataHolder('phishing', 0, lam)
inner_loop = 30
outer_epoch = 2
omega = 0
gamma = 1
eta_0 = 0.1
size_b = 4
size_bh = 25


# RBB1
w_ref = np.array([0.]*holder.dim)
S1 = []
D1 = []
SS1 = []
E1 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    D1.append(omega)
    S1.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SS1.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D1.append(omega)
    S1.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), size_b)
        v = holder.logistic_indiv_grad(omega, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + phi
        batch_bh = random.sample(list(range(holder.num)), size_bh)
        s = omega - omega_old
        y = holder.logistic_indiv_grad(omega, batch_bh) - holder.logistic_indiv_grad(omega_old, batch_bh)
        ita1 = (1 / size_bh) * gamma * (np.linalg.norm(s, ord=2)**2) / s.dot(y)
        if k >= 2:
            E1.append(ita1)
        # ita2 = (1 / size_bh) * gamma * (s.dot(y)) / (np.linalg.norm(y, ord=2) ** 2)
        # ita = np.sqrt(ita1*ita2)
        # ita = ita1 * 0.8 + ita2 * 0.2
        omega_old = omega
        omega = omega_old - ita1 * v
        D1.append(omega)
        S1.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D1[-1]
    D1 = []


# RBB2
w_ref = np.array([0.]*holder.dim)
S2 = []
D2 = []
SS2 = []
E2 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    D2.append(omega)
    S2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SS2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D2.append(omega)
    S2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), size_b)
        v = holder.logistic_indiv_grad(omega, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + phi
        batch_bh = random.sample(list(range(holder.num)), size_bh)
        s = omega - omega_old
        y = holder.logistic_indiv_grad(omega, batch_bh) - holder.logistic_indiv_grad(omega_old, batch_bh)
        ita2 = (1 / size_bh) * gamma * (s.dot(y)) / (np.linalg.norm(y, ord=2) ** 2)
        if k >= 2:
            E2.append(ita2)
        # ita = np.sqrt(ita1*ita2)
        # ita = ita1 * 0.8 + ita2 * 0.2
        omega_old = omega
        omega = omega_old - ita2 * v
        D2.append(omega)
        S2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D2[-1]
    D2 = []


# BB1
w_ref = np.array([0.]*holder.dim)
S3 = []
D3 = []
SS3 = []
W = []
E3 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    W.append(w_ref)
    omega = w_ref
    omega_old = omega
    D3.append(omega)
    S3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SS3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    if k > 1:
        eta = size_b/inner_loop * (np.linalg.norm(W[-1]-W[-2], ord=2)**2/(W[-1]-W[-2]).dot(holder.logistic_indiv_grad(W[-1])-holder.logistic_indiv_grad(W[-2])))
    else:
        eta = eta_0
    omega = omega_old - eta * v
    D3.append(omega)
    S3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), size_b)
        v = holder.logistic_indiv_grad(omega, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + phi
        if k >= 2:
            E3.append(eta)
        omega = omega_old - eta * v
        D3.append(omega)
        S3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D3[-1]
    D3 = []


# sns.set(style='darkgrid')
plt.figure()
plt.xlabel('Number of Inner Iterations within the 2nd Epoch')
plt.ylabel(r'Step Sizes')
plt.gcf().subplots_adjust(left=0.14)
epochs = [int(i) for i in range(1, (outer_epoch-1)*inner_loop+1)]
# the constants below are used for computing the effective passes with: constant = 1 + 2 * size_b * inner_loop / n
pass1 = 1.005*np.array([int(i) for i in range(1, 21)])
pass2 = 1.002*np.array([int(i) for i in range(1, 21)])
pass3 = 1.002*np.array([int(i) for i in range(1, 21)])
pass4 = 1.01*np.array([int(i) for i in range(1, 21)])
pass5 = 1.015*np.array([int(i) for i in range(1, 21)])
# plt.xlim(0, 50)
plt.xticks(range(0, inner_loop+1, 5))
# plt.ylim(10**(-14), 10**(-2))
line1, = plt.semilogy(epochs, E1, linestyle='-.', linewidth=2, color='steelblue', marker='o', markersize=6, label=r'mS2GD-RBB1')
line2, = plt.semilogy(epochs, E2, linestyle='--', linewidth=2, color='rosybrown', marker='o', markersize=6, label=r'mS2GD-RBB2')
line3, = plt.semilogy(epochs, E3, linestyle='-', linewidth=2, color='orange', marker='*', markersize=6, label=r'mS2GD-BB1')
# line4, = plt.semilogy(epochs, E4, linestyle='-', linewidth=2.5, color='mediumpurple', label=r'MB-holder-RCBB(12)  $b=4$ $b_H=40$ $\gamma=1$')
# line5, = plt.semilogy(pass5, SSC4, linestyle='-', linewidth=2.5, color='silver', label=r'MB-holder-RCBB(13)  $b=4$ $b_H=40$ $\gamma=1$')
font1 = {'size': 7}
plt.legend(handles=[line1, line2, line3], prop=font1)
plt.grid(True)
# plt.savefig('BB1_RBB_phishing_mS2GD.eps', dpi=600, format='eps')
plt.show()
