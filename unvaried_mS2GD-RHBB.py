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
inner_loop = 35
outer_epoch = 20
omega = 0
eta_0 = 0.1
size_b = 4
size_bh = 40
size_bh2 = 40
h = max(size_bh, size_bh2)
gamma2 = 1

# RBB
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
        ita1 = (1 / size_bh) * (np.linalg.norm(s, ord=2)**2) / s.dot(y)
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


# RHBB
w_ref = np.array([0.]*holder.dim)
SC = []
DC = []
SSC = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    DC.append(omega)
    SC.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SSC.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    DC.append(omega)
    SC.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
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
        ita = ita1 * 10 + ita2 * (-9)
        omega_old = omega
        omega = omega_old - ita * v
        DC.append(omega)
        SC.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = DC[-1]
    DC = []

# RHBB
w_ref = np.array([0.]*holder.dim)
SC2 = []
DC2 = []
SSC2 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    DC2.append(omega)
    SC2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SSC2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    DC2.append(omega)
    SC2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
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
        ita = ita1 * 11 + ita2 * (-10)
        omega_old = omega
        omega = omega_old - ita * v
        DC2.append(omega)
        SC2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = DC2[-1]
    DC2 = []


# RHBB
w_ref = np.array([0.]*holder.dim)
SC3 = []
DC3 = []
SSC3 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    DC3.append(omega)
    SC3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SSC3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    DC3.append(omega)
    SC3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
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
        ita = ita1 * 12 + ita2 * (-11)
        omega_old = omega
        omega = omega_old - ita * v
        DC3.append(omega)
        SC3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = DC3[-1]
    DC3 = []

# RHBB
w_ref = np.array([0.]*holder.dim)
SC4 = []
DC4 = []
SSC4 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    DC4.append(omega)
    SC4.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SSC4.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    DC4.append(omega)
    SC4.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
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
        ita = ita1 * 13 + ita2 * (-12)
        omega_old = omega
        omega = omega_old - ita * v
        DC4.append(omega)
        SC4.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = DC4[-1]
    DC4 = []


plt.figure()
plt.xlabel('Number of Effective Passes')
plt.ylabel(r'$||\nabla P(w)||^2$')
epochs = [int(i) for i in range(1, outer_epoch+1)]
# the constants below are used for computing the effective passes with: constant = 1 + 2 * size_b * inner_loop / n
pass1 = 1.006*np.array([int(i) for i in range(1, 21)])
pass2 = 1.003*np.array([int(i) for i in range(1, 21)])
pass3 = 1.003*np.array([int(i) for i in range(1, 21)])
pass4 = 1*np.array([int(i) for i in range(1, 21)])
pass5 = 1.01*np.array([int(i) for i in range(1, 21)])
pass6 = 1.02*np.array([int(i) for i in range(1, 21)])
# plt.xlim(0, 50)
plt.xticks(range(0, outer_epoch+1, 5))
plt.ylim(10**(-14), 10**(-2))
line1, = plt.semilogy(pass1, SS1, linestyle='--', linewidth=2.5, color='brown', label=r'mS2GD-RBB $b=4$ $b_H=40$ $\gamma_2=1$')
line2, = plt.semilogy(pass1, SSC, linestyle='-', linewidth=2.5, color='pink', label=r'mS2GD-RHBB(10)  $b=4$ $b_H=40$ $\gamma_2=1$')
line3, = plt.semilogy(pass1, SSC2, linestyle='-', linewidth=2.5, color='aqua', label=r'mS2GD-RHBB(11)  $b=4$ $b_H=40$ $\gamma_2=1$')
line4, = plt.semilogy(pass1, SSC3, linestyle='-', linewidth=2.5, color='mediumslateblue', label=r'mS2GD-RHBB(12)  $b=4$ $b_H=40$ $\gamma_2=1$')
line5, = plt.semilogy(pass1, SSC4, linestyle='-', linewidth=2.5, color='gray', label=r'mS2GD-RHBB(13)  $b=4$ $b_H=40$ $\gamma_2=1$')
font1 = {'size': 7}
plt.legend(handles=[line1, line2, line3, line4, line5], prop=font1)
# plt.savefig('a8a_s_all2.eps', dpi=600, format='eps')
plt.show()
