import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

from scipy.stats.mstats import normaltest

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import optuna

import tensorflow as tf


# Loading Data

data0 = pd.read_csv('../input/ecommerce-customers/Ecommerce Customers.csv')
data0.head()

print('Number of rows:', data0.shape[0])
print('Number of columns:', data0.shape[1])

# Checking For Missing Values

data0.isna().sum()

data1 = data0.drop(['Email', 'Address', 'Avatar'], axis=1)
data1.describe()


#EDA
plt.figure(figsize= (15, 5))
sns.heatmap(data1.corr(), annot= True, cmap='YlGnBu')
plt.show()

data1 = data1.drop('Time on Website', axis=1)

sns.histplot(data1['Yearly Amount Spent'], kde= True, color = '#3e236f');
normaltest(data1['Yearly Amount Spent']).pvalue.round(3)

#Distribution of Yearly Amount Spent seems to be normal, so I don't think there is a need to apply log transformation to it

grid = sns.PairGrid(data1, vars=['Avg. Session Length', 'Time on App', 'Length of Membership', 'Yearly Amount Spent'],
                    height=2, aspect = 2)

grid = grid.map_diag(plt.hist)
grid = grid.map_lower(sns.regplot, scatter_kws = {'s': 15, 'alpha': 0.7, 'color': '#005b96'}, 
                      line_kws = {'color':'orange', 'linewidth': 2})
grid = grid.map_upper(sns.kdeplot, n_levels = 10, cmap= 'coolwarm', shade = True)

plt.show()

#There is definitely a relationship between our target variable and length of membership and also all of our features and our target variable have a bell shaped curve

fig, axs = plt.subplots(2, 2, figsize= (20, 10))
fig.suptitle('Scatterplot and Boxplot of Time on App and Length of Membership', fontsize = 20)

sns.regplot(x = data1['Length of Membership'], y= data1['Yearly Amount Spent'], scatter_kws = {'s': 20, 'color': '#005b96', 'alpha': 0.7}, line_kws = {'linewidth': 2, 'color': 'orange'}, ax = axs[0, 0])
sns.boxplot(x = data1['Length of Membership'], ax= axs[0, 1])

sns.regplot(x = data1['Time on App'], y= data1['Yearly Amount Spent'], scatter_kws = {'s': 20, 'color': '#005b96','alpha': 0.7}, line_kws = {'linewidth': 2, 'color': 'orange'}, ax = axs[1, 0])
sns.boxplot(x = data1['Time on App'], ax= axs[1, 1])

plt.show()

# Scaling and Splitting

X = data2.drop(['Yearly Amount Spent'], axis=1)
y = data2['Yearly Amount Spent']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)


# Hyperparameter optimization

def random_forest_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 600)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    bootstrap = trial.suggest_categorical('bootstrap', ['True', 'False'])

    model = RandomForestRegressor(
        n_estimators= n_estimators,
        max_depth= max_depth,
        bootstrap= bootstrap
    )

    model.fit(X_train, y_train)
    cv_score = - cross_val_score(model, X_train, y_train, scoring= 'neg_mean_squared_error', cv= kf)

    return np.mean(cv_score)

study = optuna.create_study(direction= 'minimize')
study.optimize(random_forest_objective, n_trials= 100)


def catboost_objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0, 0.5)
    iterations = trial.suggest_int('iterations', 50, 300)
    depth = trial.suggest_int('depth', 3, 10)

    model = CatBoostRegressor(
        learning_rate= learning_rate,
        iterations= iterations,
        depth= depth,
        verbose= 0
    )

    model.fit(X_train, y_train)
    cv_score = - cross_val_score(model, X_train, y_train, scoring= 'neg_mean_squared_error', cv= kf)

    return np.mean(cv_score)

study = optuna.create_study(direction= 'minimize')
study.optimize(catboost_objective, n_trials= 100)

def ridge_objective(trial):
    tol = trial.suggest_loguniform('tol', 1e-7, 0.1)
    alpha = trial.suggest_float('alpha', 0, 1)

    model = Ridge(
        tol= tol,
        alpha= alpha
    )

    model.fit(X_train, y_train)
    cv_score = - cross_val_score(model, X_train, y_train, scoring= 'neg_mean_squared_error', cv= kf)

    return np.mean(cv_score)

study = optuna.create_study(direction= 'minimize')
study.optimize(ridge_objective, n_trials= 100)

def lasso_objective(trial):
    tol = trial.suggest_loguniform('tol', 1e-7, 0.1)
    alpha = trial.suggest_float('alpha', 0, 1),

    model = Lasso(
        tol= tol,
        alpha= alpha
    )

    model.fit(X_train, y_train)
    cv_score = - cross_val_score(model, X_train, y_train, scoring= 'neg_mean_squared_error', cv= kf)

    return np.mean(cv_score)

study = optuna.create_study(direction= 'minimize')
study.optimize(lasso_objective, n_trials= 100)

def knn_objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
    leaf_size = trial.suggest_int('leaf_size', 1, 50)

    model = KNeighborsRegressor(
        n_neighbors= n_neighbors,
        leaf_size= leaf_size
    )

    model.fit(X_train, y_train)
    cv_score = - cross_val_score(model, X_train, y_train, scoring= 'neg_mean_squared_error', cv= kf)

    return np.mean(cv_score)

study = optuna.create_study(direction= 'minimize')
study.optimize(knn_objective, n_trials= 100)

from tabnanny import verbose


random_forest_params = {
    'n_estimators': 479, 
    'max_depth': 18, 
    'bootstrap': 'False'
}

catboost_params = {
    'learning_rate': 0.09283646947015993, 
    'iterations': 286, 
    'depth': 3,
    'verbose': 0
}

ridge_params = {
    'tol': 2.6773399620833603e-07, 
    'alpha': 0.058158941739886394
}

lasso_params = {
    'tol': 0.0748529183574545, 
    'alpha': 0.03140815868349302
}

knn_params = {
    'n_neighbors': 4, 
    'leaf_size': 33
}


# Modelling

models = {
    'random forest': RandomForestRegressor(**random_forest_params),
    'catboost' : CatBoostRegressor(**catboost_params),
    'ridge' : Ridge(**ridge_params),
    'lasso' : Lasso(**lasso_params),
    'knn' : KNeighborsRegressor(**knn_params)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name + " trained")

# Evaluation

results = {}

kf = KFold(n_splits= 10)

for name, model in models.items():
    result = np.sqrt(-cross_val_score(model, X_train, y_train, scoring = 'neg_mean_squared_error', cv= kf))
    results[name] = result

for name, result in results.items():
    print("-------\n" + name)
    print(np.mean(result))
    print(np.std(result))


# Final Predictions

final_predictions = (
    0.2 * models['catboost'].predict(X_test) +
    0.4 * models['ridge'].predict(X_test) +
    0.4 * models['lasso'].predict(X_test) 
)

r2 = r2_score(y_test, final_predictions)
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))

print("RMSE:", rmse)
print("R-square:", r2)

# Visualising results

# Distribution of error
sns.histplot(y_test - final_predictions, color = '#005b96', kde= True)
plt.xlabel('Error')
normaltest(final_predictions).pvalue.round(3)


plt.figure(figsize= (10, 6))
sns.scatterplot(x= y_test, y= final_predictions, color= '#005b96')
plt.xlabel('Actual Yearly Amount Spent')
plt.ylabel('Predicted Yearly Amount Spent')
plt.show()


plt.figure(figsize= (10, 6))
sns.residplot(x= y_test, y = final_predictions, color= '#005b96')
plt.show()

# ANN

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(80, activation = 'relu'),
    tf.keras.layers.Dense(80, activation = 'relu'),
    tf.keras.layers.Dense(80, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)

model.compile(
    loss = tf.keras.losses.mae,
    optimizer = 'adam',
    metrics = 'mae'
)

history = model.fit(X_train, y_train, validation_split = 0.10, epochs = 100)

history.history.keys()

plt.figure(figsize=(10, 6))

epochs = range(1, 101)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)

print('R-square:', r2_score(y_test, y_pred))