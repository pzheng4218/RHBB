from libsvm.python.svmutil import *
import numpy as np
import pickle
from sklearn.preprocessing import normalize
import random
import matplotlib.pyplot as plt


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
holder = DataHolder('a8a', 0, lam)
omega = 0
eta_0 = 1


# mS2GD-RHBB(3)
size_b = 4
size_bh = 40
size_bh2 = 40
h = max(size_bh, size_bh2)
gamma2 = 1
inner_loop = 100
outer_epoch = 20
w_ref = np.array([0.]*holder.dim)
S1 = []
D1 = []
SS1 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    D1.append(omega)
    S1.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SS1.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D1.append(omega)
    S1.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), size_b)
        v = holder.logistic_indiv_grad(omega, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + phi
        batch_bh = random.sample(list(range(holder.num)), size_bh)
        batch_bh2 = random.sample(list(range(holder.num)), size_bh2)
        s = omega - omega_old
        y1 = holder.logistic_indiv_grad(omega, batch_bh) - holder.logistic_indiv_grad(omega_old, batch_bh)
        ita1 = (1 / h) * gamma2 * (np.linalg.norm(s, ord=2) ** 2) / s.dot(y1)
        y2 = holder.logistic_indiv_grad(omega, batch_bh2) - holder.logistic_indiv_grad(omega_old, batch_bh2)
        ita2 = (1 / h) * gamma2 * (s.dot(y2)) / (np.linalg.norm(y2, ord=2) ** 2)
        # ita = np.sqrt(ita1*ita2)
        ita = ita1 * 3 + ita2 * (-2)
        omega_old = omega
        omega = omega_old - ita * v
        D1.append(omega)
        S1.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D1[-1]
    D1 = []


# MB-SARAH-RHBB(3)
size_b = 4
size_bh = 40
size_bh2 = 40
h = max(size_bh, size_bh2)
gamma = 1
inner_loop = 100
outer_epoch = 20
w_ref = np.array([0.]*holder.dim)
S2 = []
D2 = []
SS2 = []
for k in range(1, outer_epoch+1):
    omega = w_ref
    omega_old = omega
    D2.append(omega)
    S2.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    SS2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D2.append(omega)
    S2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), size_b)
        v = holder.logistic_indiv_grad(omega, batch_b) - holder.logistic_indiv_grad(omega_old, batch_b) + v
        batch_bh = random.sample(list(range(holder.num)), size_bh)
        batch_bh2 = random.sample(list(range(holder.num)), size_bh2)
        s = omega - omega_old
        y1 = holder.logistic_indiv_grad(omega, batch_bh) - holder.logistic_indiv_grad(omega_old, batch_bh)
        ita1 = (1 / h) * gamma * (np.linalg.norm(s, ord=2) ** 2) / s.dot(y1)
        y2 = holder.logistic_indiv_grad(omega, batch_bh2) - holder.logistic_indiv_grad(omega_old, batch_bh2)
        ita2 = (1 / h) * gamma * (s.dot(y2)) / (np.linalg.norm(y2, ord=2) ** 2)
        # ita = np.sqrt(ita1*ita2)
        ita = ita1 * 3 + ita2 * (-2)
        omega_old = omega
        omega = omega_old - ita * v
        D2.append(omega)
        S2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D2[-1]
    D2 = []


# SVRG(1)
inner_loop = 120
outer_epoch = 20
w_ref = np.array([0.]*holder.dim)
S3 = []
D3 = []
SS3 = []
eta = 1
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    D3.append(omega)
    S3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SS3.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta * v
    D3.append(omega)
    S3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), 1)
        v = holder.logistic_indiv_grad(omega, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + phi
        omega_old = omega
        omega = omega_old - eta * v
        D3.append(omega)
        S3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D3[-1]
    D3 = []


# SVRG-BB
inner_loop = 35
outer_epoch = 20
w_ref = np.array([0.]*holder.dim)
w_ref_old = np.array([0.]*holder.dim)
eta = 0
S4 = []
D4 = []
SS4 = []
for k in range(1, outer_epoch+1):
    if k > 1:
        eta = (1/inner_loop) * (np.linalg.norm(w_ref - w_ref_old, ord=2)**2 / (w_ref - w_ref_old).dot(holder.logistic_indiv_grad(w_ref)-holder.logistic_indiv_grad(w_ref_old)))
    omega = w_ref
    omega_old = omega
    D4.append(omega)
    S4.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    SS4.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), 1)
        v = holder.logistic_indiv_grad(omega_old, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + holder.logistic_indiv_grad(w_ref)
        if k == 1:
            omega = omega_old - eta_0 * v
        else:
            omega = omega_old - eta * v
        omega_old = omega
        D4.append(omega)
        S4.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2) ** 2)
    w_ref_old = w_ref
    w_ref = D4[-1]
    D4 = []


# mS2GD-BB
size_b = 4
inner_loop = 40
outer_epoch = 20
w_ref = np.array([0.]*holder.dim)
w_ref_old = np.array([0.]*holder.dim)
eta = 0
S5 = []
D5 = []
SS5 = []
for k in range(1, outer_epoch+1):
    if k > 1:
        eta = (size_b/inner_loop) * (np.linalg.norm(w_ref - w_ref_old, ord=2)**2 / (w_ref - w_ref_old).dot(holder.logistic_indiv_grad(w_ref)-holder.logistic_indiv_grad(w_ref_old)))
    omega = w_ref
    omega_old = omega
    D5.append(omega)
    S5.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    SS5.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    m = random.randint(1, inner_loop)
    for j in range(1, m+1):
        batch_b = random.sample(list(range(holder.num)), size_b)
        v = holder.logistic_indiv_grad(omega_old, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + holder.logistic_indiv_grad(w_ref)
        if k == 1:
            omega = omega_old - eta_0 * v
        else:
            omega = omega_old - eta * v
        omega_old = omega
        D5.append(omega)
        S5.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2) ** 2)
    w_ref_old = w_ref
    w_ref = D5[-1]
    D5 = []


# Acc-Prox-SVRG-BB
size_b = 100
inner_loop = 100
outer_epoch = 15
sigma = 0.4
beta = (size_b - 2) / (size_b + 2)
w_ref = np.array([0.]*holder.dim)
w_ref_old = np.array([0.]*holder.dim)
S6 = []
D6 = []
SS6 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    if k > 1:
        eta = (sigma / inner_loop) * (np.linalg.norm(w_ref - w_ref_old, ord=2) ** 2 / (w_ref - w_ref_old).dot(
            holder.logistic_indiv_grad(w_ref) - holder.logistic_indiv_grad(w_ref_old)))
    omega = w_ref
    y = w_ref
    omega_old = omega
    D6.append(omega)
    S6.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SS6.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), size_b)
        v = holder.logistic_indiv_grad(y, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + phi
        omega_old = omega
        if k == 1:
            omega = y - eta_0 * v
        else:
            omega = y - eta * v
        y = omega + beta * (omega - omega_old)
        D6.append(omega)
        S6.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D6[-1]
    D6 = []


# Acc-Prox-SVRG
size_b = 100
inner_loop = 100
outer_epoch = 15
beta = (size_b - 2) / (size_b + 2)
w_ref = np.array([0.]*holder.dim)
S7 = []
D7 = []
SS7 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    y = w_ref
    omega_old = omega
    D7.append(omega)
    S7.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SS7.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = y - eta_0 * v
    y = omega + beta * (omega - omega_old)
    D7.append(omega)
    S7.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), size_b)
        v = holder.logistic_indiv_grad(y, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + phi
        omega_old = omega
        omega = y - eta * v
        y = omega + beta * (omega - omega_old)
        D7.append(omega)
        S7.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D7[-1]
    D7 = []


# SARAH+
size_b = 4
inner_loop = 35
outer_epoch = 20
alpha = 3
gamma = 0.125
w_ref = np.array([0.]*holder.dim)
S8 = []
D8 = []
SS8 = []
for k in range(1, outer_epoch+1):
    omega = w_ref
    omega_old = omega
    D8.append(omega)
    S8.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2) ** 2)
    SS8.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2) ** 2)
    v = holder.logistic_indiv_grad(w_ref)
    v0 = v
    omega = omega_old - alpha * v
    D8.append(omega)
    S8.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2) ** 2)
    j = 1
    while np.linalg.norm(v, ord=2) > gamma * np.linalg.norm(v0, ord=2) and j <= inner_loop:
        batch_p = random.sample(list(range(holder.num)), size_b)
        v = holder.logistic_indiv_grad(omega, batch_p) - holder.logistic_indiv_grad(omega_old, batch_p) + v
        omega_old = omega
        omega = omega_old - alpha * v
        D8.append(omega)
        S8.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2) ** 2)
        j += 1
    w_ref = D8[-1]
    D8 = []


# SVRG-ABB
size_b = 4
inner_loop = 35
outer_epoch = 20
kappa = 0.5
w_ref = np.array([0.]*holder.dim)
w_ref_old = np.array([0.]*holder.dim)
eta = 0
S9 = []
D9 = []
SS9 = []
for k in range(1, outer_epoch+1):
    if k > 1:
        eta1 = (1/inner_loop) * (np.linalg.norm(w_ref - w_ref_old, ord=2)**2 / (w_ref - w_ref_old).dot(holder.logistic_indiv_grad(w_ref)-holder.logistic_indiv_grad(w_ref_old)))
        eta2 = (1/inner_loop) * (w_ref - w_ref_old).dot(holder.logistic_indiv_grad(w_ref)-holder.logistic_indiv_grad(w_ref_old)) / (np.linalg.norm(holder.logistic_indiv_grad(w_ref) - holder.logistic_indiv_grad(w_ref_old), ord=2)**2)
        if eta2 / eta1 < kappa:
            eta = eta2
        else:
            eta = eta1
    omega = w_ref
    omega_old = omega
    D9.append(omega)
    S9.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    SS9.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), 1)
        v = holder.logistic_indiv_grad(omega_old, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + holder.logistic_indiv_grad(w_ref)
        if k == 1:
            omega = omega_old - eta_0 * v
        else:
            omega = omega_old - eta * v
        omega_old = omega
        D9.append(omega)
        S9.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2) ** 2)
    w_ref_old = w_ref
    w_ref = D9[-1]
    D9 = []


plt.figure()
plt.xlabel('Number of Effective Passes')
plt.ylabel(r'$||\nabla P(w)||^2$')
# the constants below are used for computing the effective passes with: constant = 1 + 2 * size_b * inner_loop / n
pass1 = np.array([int(i) for i in range(1, 21)])
pass2 = np.array([int(i) for i in range(1, 21)])
pass3 = np.array([int(i) for i in range(1, 21)])
pass4 = np.array([int(i) for i in range(1, 21)])
pass5 = np.array([int(i) for i in range(1, 21)])
pass6 = 1.44 * np.array([int(i) for i in range(1, 16)])
pass7 = 1.44 * np.array([int(i) for i in range(1, 16)])
pass8 = np.array([int(i) for i in range(1, 21)])
pass9 = np.array([int(i) for i in range(1, 21)])
plt.xlim(0, 16)
plt.xticks(range(0, 16, 3))
plt.ylim(10**(-14), 10**(-2))
line1, = plt.semilogy(pass1, SS1, linestyle='-', linewidth=2, color='black', label=r'mS2GD-RHBB(3)')
line2, = plt.semilogy(pass2, SS2, linestyle='-', linewidth=2, color='lightpink', label=r'MB-SARAH-RHBB(3)')
line3, = plt.semilogy(pass3, SS3, linestyle='--', linewidth=2, color='cyan', label=r'SVRG')
line4, = plt.semilogy(pass4, SS4, linestyle='--', linewidth=2, color='mediumpurple', label=r'SVRG-BB')
line5, = plt.semilogy(pass5, SS5, linestyle='-', linewidth=2, color='silver', label=r'mS2GD-BB')
line6, = plt.semilogy(pass6, SS6, linestyle='-', linewidth=2, color='darkorange', label=r'Acc-Prox-SVRG-BB')
line7, = plt.semilogy(pass7, SS7, linestyle='--', linewidth=2, color='olivedrab', label=r'Acc-Prox-SVRG')
line8, = plt.semilogy(pass8, SS8, linestyle=':', linewidth=2, color='peru', label=r'SARAH+')
line9, = plt.semilogy(pass9, SS9, linestyle=':', linewidth=2, color='steelblue', label=r'SVRG-ABB')
font1 = {'size': 7}
plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9], prop=font1)
# plt.savefig('a8a_hexin.eps', dpi=600, format='eps')
plt.show()
