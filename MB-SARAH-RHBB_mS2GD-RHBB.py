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
eta_0 = 1
size_b = 4
size_bh = 40
size_bh2 = 40
h = max(size_bh, size_bh2)
gamma = 1
gamma2 = 1

# mS2GD-RHBB(2)
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
        ita = ita1 * 2 + ita2 * (-1)
        omega_old = omega
        omega = omega_old - ita * v
        D1.append(omega)
        S1.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D1[-1]
    D1 = []


# mS2GD-RHBB(4)
w_ref = np.array([0.]*holder.dim)
S2 = []
D2 = []
SS2 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    D2.append(omega)
    S2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SS2.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D2.append(omega)
    S2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
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
        ita = ita1 * 4 + ita2 * (-3)
        omega_old = omega
        omega = omega_old - ita * v
        D2.append(omega)
        S2.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D2[-1]
    D2 = []


# mS2GD-RHBB(6)
w_ref = np.array([0.]*holder.dim)
S3 = []
D3 = []
SS3 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    D3.append(omega)
    S3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SS3.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D3.append(omega)
    S3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    for j in range(1, inner_loop+1):
        batch_b = random.sample(list(range(holder.num)), size_b)
        v = holder.logistic_indiv_grad(omega, batch_b) - holder.logistic_indiv_grad(w_ref, batch_b) + phi
        batch_bh = random.sample(list(range(holder.num)), size_bh2)
        batch_bh2 = random.sample(list(range(holder.num)), size_bh)
        s = omega - omega_old
        y1 = holder.logistic_indiv_grad(omega, batch_bh) - holder.logistic_indiv_grad(omega_old, batch_bh)
        ita1 = (1 / h) * gamma2 * (np.linalg.norm(s, ord=2) ** 2) / s.dot(y1)
        y2 = holder.logistic_indiv_grad(omega, batch_bh2) - holder.logistic_indiv_grad(omega_old, batch_bh2)
        ita2 = (1 / h) * gamma2 * (s.dot(y2)) / (np.linalg.norm(y2, ord=2) ** 2)
        # ita = np.sqrt(ita1*ita2)
        ita = ita1 * 6 + ita2 * (-5)
        omega_old = omega
        omega = omega_old - ita * v
        D3.append(omega)
        S3.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D3[-1]
    D3 = []


# mS2GD-RHBB(8)
w_ref = np.array([0.]*holder.dim)
S4 = []
D4 = []
SS4 = []
for k in range(1, outer_epoch+1):
    phi = holder.logistic_indiv_grad(w_ref)
    omega = w_ref
    omega_old = omega
    D4.append(omega)
    S4.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    SS4.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D4.append(omega)
    S4.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
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
        ita = ita1 * 8 + ita2 * (-7)
        omega_old = omega
        omega = omega_old - ita * v
        D4.append(omega)
        S4.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D4[-1]
    D4 = []


# MB-SARAH-RHBB(2)
w_ref = np.array([0.]*holder.dim)
S5 = []
D5 = []
SS5 = []
for k in range(1, outer_epoch+1):
    omega = w_ref
    omega_old = omega
    D5.append(omega)
    S5.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    SS5.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D5.append(omega)
    S5.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
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
        ita = ita1 * 2 + ita2 * (-1)
        omega_old = omega
        omega = omega_old - ita * v
        D5.append(omega)
        S5.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D5[-1]
    D5 = []


# MB-SARAH-RHBB(4)
w_ref = np.array([0.]*holder.dim)
S6 = []
D6 = []
SS6 = []
for k in range(1, outer_epoch+1):
    omega = w_ref
    omega_old = omega
    D6.append(omega)
    S6.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    SS6.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D6.append(omega)
    S6.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
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
        ita = ita1 * 4 + ita2 * (-3)
        omega_old = omega
        omega = omega_old - ita * v
        D6.append(omega)
        S6.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D6[-1]
    D6 = []



# MB-SARAH-RHBB(6)
w_ref = np.array([0.]*holder.dim)
S7 = []
D7 = []
SS7 = []
for k in range(1, outer_epoch+1):
    omega = w_ref
    omega_old = omega
    D7.append(omega)
    S7.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2)**2)
    SS7.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D7.append(omega)
    S7.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
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
        ita = ita1 * 6 + ita2 * (-5)
        omega_old = omega
        omega = omega_old - ita * v
        D7.append(omega)
        S7.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2)**2)
    # w_ref = random.choice(D)
    w_ref = D7[-1]
    D7 = []

# MB-SARAH-RHBB(8)
w_ref = np.array([0.] * holder.dim)
S8 = []
D8 = []
SS8 = []
for k in range(1, outer_epoch + 1):
    omega = w_ref
    omega_old = omega
    D8.append(omega)
    S8.append(np.linalg.norm(holder.logistic_indiv_grad(w_ref), ord=2) ** 2)
    SS8.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2) ** 2)
    v = holder.logistic_indiv_grad(w_ref)
    omega = omega_old - eta_0 * v
    D8.append(omega)
    S8.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2) ** 2)
    for j in range(1, inner_loop + 1):
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
        ita = ita1 * 8 + ita2 * (-7)
        omega_old = omega
        omega = omega_old - ita * v
        D8.append(omega)
        S8.append(np.linalg.norm(holder.logistic_indiv_grad(omega), ord=2) ** 2)
    # w_ref = random.choice(D)
    w_ref = D8[-1]
    D8 = []





plt.figure()
plt.xlabel('Number of Effective Passes')
plt.ylabel(r'$||\nabla P(w)||^2$')
epochs = [int(i) for i in range(1, outer_epoch+1)]
# the constants below are used for computing the effective passes with: constant = 1 + 2 * size_b * inner_loop / n
# plt.xlim(0, 20)
pass1 = 1.006*np.array([int(i) for i in range(1, outer_epoch+1)])
pass2 = 1.003*np.array([int(i) for i in range(1, outer_epoch+1)])
pass3 = 1.003*np.array([int(i) for i in range(1, outer_epoch+1)])
pass4 = 1*np.array([int(i) for i in range(1, outer_epoch+1)])
pass5 = 1.01*np.array([int(i) for i in range(1, outer_epoch+1)])
pass6 = 1.02*np.array([int(i) for i in range(1, outer_epoch+1)])
plt.xticks(range(0, outer_epoch+1, 5))
plt.ylim(10**(-14), 10**(-2))
line1, = plt.semilogy(pass1, SS1, linestyle='--', linewidth=2, color='brown', label=r'mS2GD-RHBB(2)')
line2, = plt.semilogy(pass1, SS2, linestyle='--', linewidth=2, color='lightpink', label=r'mS2GD-RHBB(4)')
line3, = plt.semilogy(pass1, SS3, linestyle='--', linewidth=2, color='cyan', label=r'mS2GD-RHBB(6)')
line4, = plt.semilogy(pass1, SS4, linestyle='--', linewidth=2, color='mediumpurple', label=r'mS2GD-RHBB(8)')
line5, = plt.semilogy(pass1, SS5, linestyle='-', linewidth=2, color='silver', label=r'MB-SARAH-RHBB(2)')
line6, = plt.semilogy(pass1, SS6, linestyle='-', linewidth=2, color='darkorange', label=r'MB-SARAH-RHBB(4)')
line7, = plt.semilogy(pass1, SS7, linestyle='-', linewidth=2, color='olivedrab', label=r'MB-SARAH-RHBB(6)')
line8, = plt.semilogy(pass1, SS8, linestyle='-', linewidth=2, color='orchid', label=r'MB-SARAH-RHBB(8)')

font1 = {'size': 7}
plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8], prop=font1)
# plt.savefig('a8a_shuang.eps', dpi=600, format='eps')
plt.show()
