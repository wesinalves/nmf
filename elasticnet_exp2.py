import numpy as np
from sklearn.linear_model import ElasticNet

##################################################
# Generate some sparce data to play with
n_samples_train, n_samples_test, n_features = 75,150,500
np.random.seed(0)
coef = np.random.randn(n_features)
coef[50:] = 0.0
X = np.random.randn(n_samples_train + n_samples_test, n_features)
y = np.dot(X,coef)

X_train, y_train = X[:n_samples_train],y[:n_samples_train]
X_test, y_test = X[n_samples_train:], y[n_samples_train:]

#compute train and test errors
alphas = np.logspace(-5,1,60)
enet = ElasticNet(l1_ratio = 0.7)

train_errors = list()
test_errors = list()

for alpha in alphas:
	enet.set_params(alpha=alpha)
	enet.fit(X_train, y_train)
	train_errors.append(enet.score(X_train, y_train))
	test_errors.append(enet.score(X_test, y_test))

id_alpha_optimun = np.argmax(test_errors)
alpha_optimun = alphas[id_alpha_optimun]
print("Optimal regularization parameters : %s" % alpha_optimun)

enet.set_params(alpha=alpha_optimun)
coef_ = enet.fit(X,y).coef_

# #############################################################################
# Plot results functions

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.semilogx(alphas, train_errors, label='Train')
plt.semilogx(alphas, test_errors, label='Test')
plt.vlines(alpha_optimun, plt.ylim()[0], np.max(test_errors), color='k',
           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([0, 1.2])
plt.xlabel('Regularization params')
plt.ylabel('Performance')

# Show estimated coef_ vs true coef
plt.subplot(2, 1, 2)
plt.plot(coef, label='True coef')
plt.plot(coef_, label='Estimated coef')
plt.legend()
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)
plt.show()