import sklearn
import pandas as pd
import numpy as np
from sklearn import model_selection, ensemble
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def rocauc (estimator, X, y):
    pred = estimator.predict_proba(X)[:, 1]
    score = sklearn.metrics.roc_auc_score(y, pred)
    return score

data = pd.read_csv("data/features.csv", index_col='match_id')
print(data)
for i in range(data.shape[1]):
    if data.count()[i] != data.shape[0]:
        print(data.columns[i])

data = data.fillna(0)
data = data.sample(frac=1)
y_train = data.iloc[0:int(data.shape[0]/10), 103]
X_train = data.iloc[0:int(data.shape[0]/10), 0:102]

#градиентный бустинг

kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True)
est = np.arange(10, 110, 10)
for i in range(len(est)):
    clf = sklearn.ensemble.GradientBoostingClassifier(learning_rate=0.9, n_estimators=est[i], verbose=0, max_depth=2)
    metric_c = sklearn.model_selection.cross_val_score(clf, X_train, y_train, scoring=rocauc, cv=kf)
    print(sum(metric_c)/len(metric_c))

#логистическая регрессия

y_train = data['radiant_win']
X_train = data.iloc[:, 0:102]

X_train.index = [i for i in range(data.shape[0])]

print(X_train.nunique()[2])

# N — количество различных героев в выборке
X_pick = np.zeros((data.shape[0], X_train.nunique()[2]))

num1 = [i+1 for i in range(23)]
num2 = [i for i in range(25, 107)]
num3 = [109, 110]
num4 = [112]

for j in range(2, 75, 8):
    for i in range(data.shape[0]):
        if data.iloc[i, j] in num1:
            data.iloc[i, j] = data.iloc[i, j]
        elif data.iloc[i, j] in num2:
            data.iloc[i, j] = data.iloc[i, j] - 1
        elif data.iloc[i, j] in num3:
            data.iloc[i, j] = data.iloc[i, j] - 3
        else: data.iloc[i, j] = data.iloc[i, j] - 4

for i, match_id in enumerate(data.index):
    for p in range(5):
        X_pick[i, data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_train_drop = X_train.drop(columns=['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'])

X_pick = pd.DataFrame(X_pick)
X_train_new = X_train_drop.join(X_pick)
X_train_new.columns = X_train_new.columns.astype(str)

scaler = sklearn.preprocessing.StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_new)

grid = {'C': np.power(2.0, np.arange(-5, 5))}
logreg = sklearn.linear_model.LogisticRegression()
gs = sklearn.model_selection.GridSearchCV(logreg, grid, scoring=rocauc, cv=kf)
gs.fit(X_train_scaled, y_train)
logreg = sklearn.linear_model.LogisticRegression(C=gs.best_params_['C'])
metric_l = sklearn.model_selection.cross_val_score(logreg, X_train_scaled, y_train, scoring=rocauc, cv=kf)
print(sum(metric_l)/len(metric_l))

data_test = pd.read_csv("data/features_test.csv", index_col='match_id')
data_test = data_test.fillna(0)
X_test = data_test.iloc[:, 0:102]
X_test.index = [i for i in range(data_test.shape[0])]

X_pick = np.zeros((data_test.shape[0], X_test.nunique()[2]))

num1 = [i+1 for i in range(23)]
num2 = [i for i in range(25, 107)]
num3 = [109, 110]
num4 = [112]

for j in range(2, 75, 8):
    for i in range(data_test.shape[0]):
        if data_test.iloc[i, j] in num1:
            data_test.iloc[i, j] = data_test.iloc[i, j]
        elif data_test.iloc[i, j] in num2:
            data_test.iloc[i, j] = data_test.iloc[i, j] - 1
        elif data_test.iloc[i, j] in num3:
            data_test.iloc[i, j] = data_test.iloc[i, j] - 3
        else: data_test.iloc[i, j] = data_test.iloc[i, j] - 4

for i, match_id in enumerate(data_test.index):
    for p in range(5):
        X_pick[i, data_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_test_drop = X_test.drop(columns=['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'])
X_pick = pd.DataFrame(X_pick)
X_test_new = X_test_drop.join(X_pick)
X_test_new.columns = X_test_new.columns.astype(str)

X_test_scaled = scaler.transform(X_test_new)

logreg.fit(X_train_scaled, y_train)
#logreg.predict_proba(X_test_scaled)

pred = {'match_id': [data_test.index[i] for i in range(X_test_scaled.shape[0])], 'radiant_win': logreg.predict_proba(X_test_scaled)[:, 1]}
pred = pd.DataFrame(data=pred)
pred.set_index('match_id')