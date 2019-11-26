from math import log
import numpy as np

def hist(sx):
    d = dict()
    for s in sx:
        d[s] = d.get(s, 0) + 1
    return map(lambda z: float(z)/len(sx), d.values())

def elog(x):
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x*log(x)

def entropyfromprobs(probs, base=2):
    return -sum(map(elog, probs))/log(base)

def entropyd(sx, base=2):
    return entropyfromprobs(hist(sx), base=base)

def midd(x, y):
    return -entropyd(list(zip(x, y)))+entropyd(x)+entropyd(y)

def information_gain(f1, f2):
    ig = entropyd(f1) - conditional_entropy(f1, f2)
    return ig

def conditional_entropy(f1, f2):
    ce = entropyd(f1) - midd(f1, f2)
    return ce

def su_calculation(f1, f2):
    # calculate information gain of f1 and f2, t1 = ig(f1,f2)
    t1 = information_gain(f1, f2)
    # calculate entropy of f1, t2 = H(f1)
    t2 = entropyd(f1)
    # calculate entropy of f2, t3 = H(f2)
    t3 = entropyd(f2)
    # su(f1,f2) = 2*t1/(t2+t3)
    su = 2.0*t1/(t2+t3)
    return su

def merit_calculation(X, y):
    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su_calculation(fi, y)
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su_calculation(fi, fj)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits

def cfs(X, y):
    n_samples, n_features = X.shape
    F = []
    # M stores the merit values
    M = []
    while True:
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                # calculate the merit of current selected features
                t = merit_calculation(X[:, F], y)
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        if len(M) > 5:
            if M[len(M)-1] <= M[len(M)-2]:
                if M[len(M)-2] <= M[len(M)-3]:
                    if M[len(M)-3] <= M[len(M)-4]:
                        if M[len(M)-4] <= M[len(M)-5]:
                            break
    selectedIDX = [i+1 for i in F]
    # selectedX = X[:,F]

    return selectedIDX
