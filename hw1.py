from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale, MinMaxScaler, MaxAbsScaler, robust_scale, RobustScaler, QuantileTransformer, quantile_transform, PowerTransformer, Normalizer, KBinsDiscretizer, KBinsDiscretizer, Binarizer, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression,mutual_info_regression,chi2, f_classif, mutual_info_classif,RFE
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import cfs
import numpy as np
import pandas as pd

def NomToNum(data):
    mainhue = data[17].unique().tolist()
    topleft = data[28].unique().tolist()
    botright = data[29].unique().tolist()
    le = LabelEncoder()
    le.fit(mainhue)
    data[17] = le.transform(data[17])
    le.fit(topleft)
    data[28] = le.transform(data[28])
    le.fit(botright)
    data[29] = le.transform(data[29])
    data = data.drop([0], axis=1)
    return data

def problem2_2_1(data):
    mainhue = data[17].unique().tolist()
    topleft = data[28].unique().tolist()
    botright = data[29].unique().tolist()
    le = LabelEncoder()
    le.fit(mainhue)
    data[17] = le.transform(data[17])
    le.fit(topleft)
    data[28] = le.transform(data[28])
    le.fit(botright)
    data[29] = le.transform(data[29])
    x = data[[17,28,29]]
    ohe = OneHotEncoder()
    ohe.fit(x)
    newx = ohe.transform(x).toarray()
    for i in range(len(mainhue)):
        col_name = "mainhue_" + mainhue[i]
        data[col_name] = newx[:, i]
    for i in range(len(topleft)):
        col_name = "topleft_" + topleft[i]
        data[col_name] = newx[:, len(mainhue) + i]
    for i in range(len(botright)):
        col_name = "botright_" + botright[i]
        data[col_name] = newx[:, len(mainhue) + len(topleft) + i]
    newdata = data.drop([0,17,28,29], axis=1)
    return newdata

def problem2_2_2(data):
    landmass = data[1].unique().tolist()
    zone = data[2].unique().tolist()
    language = data[5].unique().tolist()
    religion = data[6].unique().tolist()
    x = data[[1,2,5,6]]
    ohe = OneHotEncoder()
    ohe.fit(x)
    newx = ohe.transform(x).toarray()
    for i in range(len(landmass)):
        col_name = "landmass_" + str(landmass[i])
        data[col_name] = newx[:, i]
    for i in range(len(zone)):
        col_name = "zone_" + str(zone[i])
        data[col_name] = newx[:, len(landmass) + i]
    for i in range(len(language)):
        col_name = "language_" + str(language[i])
        data[col_name] = newx[:, len(landmass) + len(zone) + i]
    for i in range(len(religion)):
        col_name = "religion_" + str(religion[i])
        data[col_name] = newx[:, len(landmass) + len(zone) + len(language) + i]
    newdata = data.drop([1,2,5,6], axis=1)
    return newdata

def problem2_3_1(area):
    x = range(len(area))
    plt.bar(x, area)
    plt.show()
    plt.hist(area)
    plt.show()

def problem2_3_2(data):
    data[3].loc[data[3] == 0] = np.nan
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(data)
    newdata_mean = imp_mean.transform(data)
    area1 = newdata_mean[:,2].tolist()
    print("Use Mean:",problem2_3_1(area1))

    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_median.fit(data)
    newdata_median = imp_median.transform(data)
    area2 = newdata_median[:,2].tolist()
    print("Use Median:", problem2_3_1(area2))

    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_freq.fit(data)
    newdata_freq = imp_freq.transform(data)
    area3 = newdata_freq[:,2].tolist()
    print("Use Most Frequent:",problem2_3_1(area3))

    imp_cons= SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=-999)
    imp_cons.fit(data)
    newdata_cons = imp_cons.transform(data)
    area4 = newdata_cons[:, 2].tolist()
    print("Use Constant:", problem2_3_1(area4))
    return "as shown in the plots"

def problem2_3_3(data):
    data[3].loc[data[3] == 0] = np.nan
    imp = IterativeImputer(missing_values=np.nan)
    imp.fit(data)
    newdata = np.round(imp.transform(data))
    area = newdata[:, 2].tolist()
    print("Use Multivariate:",problem2_3_1(area))
    return "as shown in the plots"


#problem2_4
#helper func for ploting
def plot(area):
    plt.plot(area)
    plt.ylabel('Area')
    plt.xlabel('index')
    plt.show()
    plt.hist(area)
    plt.show()


#problem2_4 i standardization
def problem2_4_1(area):
    area_scale = scale(area)
    plot(area_scale)
    return "as shown in the plots"

#problem2_4 ii scaling to a range
def problem2_4_2(area):
    MNS = MinMaxScaler(copy=True,feature_range=(0, 1)).fit_transform(area)
    print("Use MinMaxScaler:", plot(MNS))
    MAS = MaxAbsScaler().fit_transform(area)
    print("Use MaxAbsScaler:", plot(MAS))
    rbs = robust_scale(area,axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
    print("Use robust_scale:", plot(rbs))
    RBS = RobustScaler().fit_transform(area)
    print("Use RobustScaler:", plot(RBS))
    return "as shown in the plots"

#problem2_4 iii mapping to a uniform distribution
def problem2_4_3(area):
    QT= preprocessing.QuantileTransformer(random_state=0, n_quantiles = 194)
    area_QT = QT.fit_transform(area)
    print("Use QuantileTransformer:", plot(area_QT))  
    qt = quantile_transform(area, n_quantiles=194, random_state=0, copy=True)
    print("Use quantile_transform:", plot(qt))
    return "as shown in the plots"

#problem2_4 iv mapping to a Gaussian distribution
def problem2_4_4(area):
    pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
    pt = pt.fit_transform(area)
    print("Use PowerTransformer:", plot(pt))
    return "as shown in the plots"

#problem2_4 v normalization
def problem2_4_5(area):
    normalized = preprocessing.normalize(area)
    print("Use normalize:", plot(normalized))
    return "as shown in the plots"

#problem2_5 Discretization
#helper for ploting population
def plot_population(population):
    plt.plot(population)
    plt.ylabel('Population')
    plt.xlabel('index')
    plt.show()
    plt.hist(population)
    plt.show()

#plot original population
def problem2_5(population):
    plot_population(population)
    return "as shown in the plots"
    
#problem2_5 K-bins discretization
def problem2_5_1(population):
    Dis = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='quantile').fit_transform(population)
    plot_population(Dis)
    return "as shown in the plots"

#problem2_5 Feature binarization
def problem2_5_2(population):
    plot_population(Binarizer(threshold = 100).fit_transform(population))
    return "as shown in the plots"

#problem2_6 Custom transformation
def helper(a):
    return a*0.386102
def problem2_6(area):
    ct = FunctionTransformer(helper, validate = False).fit_transform(area)
    plot(ct)
    return "as shown in the plots"

####################

def problem3_1_1(data):
    datacorr = data.corr()
    plt.matshow(datacorr)
    plt.show()
    return datacorr

def problem3_1_2(data):
    datacov = data.cov()
    plt.matshow(datacov)
    plt.show()
    return datacov

def problem3_2_1(data):
    newdata = data.sample(frac=0.6, replace=False,axis=0)
    return newdata

def problem3_2_2(data):
    newdata = data.sample(frac=0.6, replace=True, axis=0)
    return newdata

def problem3_2_3(data):
    data = NomToNum(data)
    newdata = data.groupby(6).apply(lambda x: x.sample(frac=0.6,replace=False,axis=0))
    return newdata

def problem3_3_2(data):
    selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
    selector.fit_transform(data)
    newdata = data.loc[:, selector.get_support()]
    return newdata.columns,newdata

def problem3_3_3(data,method,k):
    X = NomToNum(data)
    y = X[6]
    X = X.drop([6], axis=1)
    selector = SelectKBest(method, k)
    selector.fit_transform(X, y)
    newdata =X.loc[:, selector.get_support()]
    return newdata.columns,newdata

def problem3_3_4(data,k):
    X = NomToNum(data)
    y = X[6]
    X = X.drop([6], axis=1)
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, k)
    selector.fit(X, y)
    newdata = X.loc[:, selector.support_]
    return newdata.columns,newdata

def problem3_3_5(data):
    X = NomToNum(data)
    y = X[6]
    X = X.drop([6], axis=1)
    selectedIDX = cfs.cfs(X.values, y.values)
    newdata = X.loc[:,selectedIDX]
    return selectedIDX,newdata

def problem3_4_1(data):
    pca = PCA(svd_solver='auto')
    X = NomToNum(data)
    pca.fit(X)
    print(pca.components_.shape) # 29 principal components 
    print(pca.explained_variance_.shape)#29
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)


def problem3_4_2(data):
    pca = PCA(svd_solver='auto', n_components = 1)
    X = NomToNum(data)
    p = pca.fit_transform(X)
    print(p)
    print(pca.explained_variance_ratio_)
    

data = pd.read_csv('/Users/zoey/WPI/kdd/hw1/flag.data',header=None)

# print(problem2_2_1(data))
# print(problem2_2_2(problem2_2_1(data)))

# print(problem2_3_1(data[3]))
# print(problem2_3_2(problem2_2_2(problem2_2_1(data))))
# print(problem2_3_2(NomToNum(data)))
# print(problem2_3_3(problem2_2_2(problem2_2_1(data))))
# print(problem2_3_3(NomToNum(data)))

area = data.values[:,3:4]
# print(problem2_4_1(area))
# print(problem2_4_2(area))
# print(problem2_4_3(area))
# print(problem2_4_4(area))
# print(problem2_4_5(area))

population = data.values[:,4:5]
# print(problem2_5(population))
# print(problem2_5_1(population))
# print(problem2_5_2(population))

# print(problem2_6(area))
                
# print(problem3_1_1(problem2_2_2(problem2_2_1(data))))
# print(problem3_1_1(NomToNum(data)))
# print(problem3_1_2(problem2_2_2(problem2_2_1(data))))
# print(problem3_1_2(NomToNum(data)))

# print(problem3_2_1(problem2_2_2(problem2_2_1(data))))
# print(problem3_2_1(NomToNum(data)))
# print(problem3_2_2(problem2_2_2(problem2_2_1(data))))
# print(problem3_2_2(NomToNum(data)))
# print(problem3_2_3(data))

# print(problem3_3_2(problem2_2_2(problem2_2_1(data))))
# print(problem3_3_2(NomToNum(data)))
# print(problem3_3_3(data,f_regression,k=15))
# print(problem3_3_3(data,mutual_info_regression,k=15))
# print(problem3_3_3(data,chi2,k=15))
# print(problem3_3_3(data,f_classif,k=15))
# print(problem3_3_3(data,mutual_info_classif,k=15))
# print(problem3_3_4(data,k=15))
# print(problem3_3_5(data))
# problem3_4_1(data)
# problem3_4_2(data)

