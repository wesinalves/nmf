import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

##################################################
# Generate some sparce data to play with
np.random.seed(42)

n_samples, n_features = 50,200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features,2)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0 #sparcify coef

y = np.dot(X,coef)

#add noise
#y += 0.01 * np.random.normal(size=n_samples)


#split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[:n_samples//2],y[:n_samples//2]
X_test, y_test = X[n_samples//2 :], y[n_samples//2 :]

y_train


###################################################
#ElasticNet
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train,y_train).predict(X_test)
r2_score_enet = r2_score(y_test,y_pred_enet)
print(enet)
print("r^2 on test data: %f" %r2_score_enet)