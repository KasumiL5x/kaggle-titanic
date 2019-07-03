# Please see the Jupyter notebook for discussion.

#### Load in data.
import pandas as pd
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

#### Drop unnecessary columns.
train.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)

#### Handle missing data.
train.Age.fillna(train.Age.median(), inplace=True)
test.Age.fillna(test.Age.median(), inplace=True)
#
train.Embarked.fillna(train.Embarked.mode()[0], inplace=True)
#
test.Fare.fillna(test.Fare.median(), inplace=True)

#### Family Size / Coding
train['FamilySize'] = (train.SibSp + train.Parch) + 1
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
#
test['FamilySize'] = (test.SibSp + test.Parch) + 1
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)
#
train['FamilySize'] = pd.cut(train.FamilySize, bins=[0, 2, 3, 5, 100], labels=['Single', 'Couple', 'Medium', 'Large'])
test['FamilySize'] = pd.cut(test.FamilySize, bins=[0, 2, 3, 5, 100], labels=['Single', 'Couple', 'Medium', 'Large'])

#### Cabin Coding
train['Cabin'] = pd.Series([x[0] if not pd.isnull(x) else 'X' for x in train.Cabin])
train['Cabin'].replace('T', 'X', inplace=True)
test['Cabin'] = pd.Series([x[0] if not pd.isnull(x) else 'X' for x in test.Cabin])

#### Fare Coding
train['FareCoded'] = pd.cut(train.Fare, bins=[-0.001, 12.0, 30.0, 1000.0], labels=['Third', 'Second', 'First'])
train.drop('Fare', axis=1, inplace=True)
#
test['FareCoded'] = pd.cut(test.Fare, bins=[-0.001, 12.0, 30.0, 1000.0], labels=['Third', 'Second', 'First'])
test.drop('Fare', axis=1, inplace=True)

#### Age Coding
train['AgeCoded'] = pd.cut(train.Age, bins=[0, 5, 13, 18, 65, 1000], labels=['Baby', 'Child', 'Teen', 'Adult', 'Senior'])
train.drop('Age', axis=1, inplace=True)
#
test['AgeCoded'] = pd.cut(test.Age, bins=[0, 5, 13, 18, 65, 1000], labels=['Baby', 'Child', 'Teen', 'Adult', 'Senior'])
test.drop('Age', axis=1, inplace=True)

#### Name Coding
def code_name(names):
    # Extract titles.
    titles = pd.Series([x.split(',')[1].split('.')[0].strip() for x in names])
    # Group rare entries.
    titles = titles.replace(['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    # Group non-married women (possibly, but we have limited information).
    titles = titles.replace(['Mme', 'Ms', 'Mlle'], 'Miss')
    return titles
train['Title'] = code_name(train.Name)
test['Title'] = code_name(test.Name)
#
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

#### Extract Passenger IDs
test_passenger_ids = test.PassengerId
test.drop('PassengerId', axis=1, inplace=True)

#### One-Hot Encoding
# Pandas calls these dummies. Let's do it for both sets.
train = pd.get_dummies(train)
test = pd.get_dummies(test)

#### Features and Labels
features = train.drop('Survived', axis=1)
labels = train.Survived

#### PCA
from sklearn.decomposition import PCA
n_components = 16
pca = PCA(n_components)
train_pca = pca.fit_transform(features)
test_pca = pca.transform(test)

#### Build Network
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.utils import to_categorical
def create_model(learn_rate=0.01, momentum=0, dropout_rate=0.0, weight_constraint=0):
  model = Sequential()
  model.add(Dense(20, activation='relu', input_shape=(n_components,), kernel_constraint=maxnorm(weight_constraint)))
  model.add(Dropout(dropout_rate))
  model.add(Dense(2, activation='softmax'))
  model.compile(optimizer=SGD(lr=learn_rate, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

# Refine model.
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
model_wrap = KerasClassifier(build_fn=create_model)
#! NOTE TO DANIEL: This amount of data trained for over an hour and didn't finish. Its at least 750 permutations. Tone it town a little (dropout?) or run overnight.
params = {
  #'batch_size': [10, 20, 40, 60, 80, 100],
  'validation_split': [0.2],
  'epochs': [10, 50, 80],
  #'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
  'learn_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
  #'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
  'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  'weight_constraint': [1, 2, 3, 4, 5]
}
cv = GridSearchCV(estimator=model_wrap, param_grid=params, n_jobs=-1, iid=True)
cv_results = cv.fit(train_pca, to_categorical(labels))
print('Score: {} / {}'.format(cv.best_score_, cv.best_params_))

# Extract best parameters.
best_model = cv.best_estimator_.model
# best_params = cv.best_params_

# Retrain with new parameters.
# model = create_model(learn_rate=best_params['learn_rate'])
# model_history = model.fit(train_pca, to_categorical(labels))

#model_hist = model.fit(train_pca, to_categorical(labels), epochs=80, validation_split=0.2)

#### Plot Accuracy
import matplotlib.pyplot as plt
plt.plot(best_model.history.history['loss'], label='loss')
plt.plot(best_model.history.history['val_loss'], label='val loss')
plt.legend()
plt.show()


# #### Predict & Submit
# predicted_survival = model.predict_classes(test_pca)
# print(predicted_survival)

# submission = pd.DataFrame({
#     'PassengerId': test_passenger_ids,
#     'Survived': predicted_survival
# })
# submission.to_csv('./submissions/keras-dl.csv', index=False)
# print('Done!')