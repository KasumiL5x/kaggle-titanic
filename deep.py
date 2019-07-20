import pandas as pd
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')


# `PassengerId` is highly unlikely to have any impact upon survival rate.
train.drop('PassengerId', axis=1, inplace=True)
# It's unlikely that `Ticket` numbers impacted the survival rate.
train.drop('Ticket', axis=1, inplace=True)
# Repeating for the test data, but keeping `PassengerID` as we need that!
test.drop('Ticket', axis=1, inplace=True)


# Handle `Age` in both datasets.
train.Age.fillna(train.Age.median(), inplace=True)
test.Age.fillna(test.Age.median(), inplace=True)
# Handle `Embarked` in the training data.
train.Embarked.fillna(train.Embarked.mode()[0], inplace=True)
# Handle `Fare` in the test data.
test.Fare.fillna(test.Fare.median(), inplace=True)


# Compute for test. Add 1 for self.
train['FamilySize'] = (train.SibSp + train.Parch) + 1
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
# Compute for test.
test['FamilySize'] = (test.SibSp + test.Parch) + 1
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# Replace `FamilySize` with coded version.
train['FamilySize'] = pd.cut(train.FamilySize, bins=[0, 2, 3, 5, 100], labels=['Single', 'Couple', 'Medium', 'Large'])
# Do the same for test.
test['FamilySize'] = pd.cut(test.FamilySize, bins=[0, 2, 3, 5, 100], labels=['Single', 'Couple', 'Medium', 'Large'])

# Cabin coding.
train['Cabin'] = pd.Series([x[0] if not pd.isnull(x) else 'X' for x in train.Cabin])
# Fix rogue cabin.
train['Cabin'].replace('T', 'X', inplace=True)
test['Cabin'] = pd.Series([x[0] if not pd.isnull(x) else 'X' for x in test.Cabin])


# Cut the `Fare` into groups roughly matching the above.
# -0.001 is used as floats cannot be matched directly to 0 due to imprecision.
# 1000 is used as an arbitrary max to encompass all. This could also be max(Fare) + a bit.
# Boundaries are set at 12 for third class, even though *technically* 8 pounds is the max.
train['FareCoded'] = pd.cut(train.Fare, bins=[-0.001, 12.0, 30.0, 1000.0], labels=['Third', 'Second', 'First'])
train.drop('Fare', axis=1, inplace=True)
# Repeat for test dataset.
test['FareCoded'] = pd.cut(test.Fare, bins=[-0.001, 12.0, 30.0, 1000.0], labels=['Third', 'Second', 'First'])
test.drop('Fare', axis=1, inplace=True)
# Drop `Pclass` to avoid redundant conflict between the fare (which indicates class *mostly* and the actual passenger class).
train.drop('Pclass', axis=1, inplace=True)
test.drop('Pclass', axis=1, inplace=True)


# Cut the `Age` into groups as above.
# A max age of 1000 is used to to capture everything; it's value doesn't really matter.
train['AgeCoded'] = pd.cut(train.Age, bins=[0, 5, 13, 18, 65, 1000], labels=['Baby', 'Child', 'Teen', 'Adult', 'Senior'])
train.drop('Age', axis=1, inplace=True)
# Repeat for test dataset.
test['AgeCoded'] = pd.cut(test.Age, bins=[0, 5, 13, 18, 65, 1000], labels=['Baby', 'Child', 'Teen', 'Adult', 'Senior'])
test.drop('Age', axis=1, inplace=True)


def code_name(names):
    # Extract titles.
    titles = pd.Series([x.split(',')[1].split('.')[0].strip() for x in names])
    # Group rare entries.
    titles = titles.replace(['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    # Group non-married women (possibly, but we have limited information).
    titles = titles.replace(['Mme', 'Ms', 'Mlle'], 'Miss')
    return titles
# Make `Title` column from coded titles.
train['Title'] = code_name(train.Name)
test['Title'] = code_name(test.Name)
# Drop original `Name` columns.
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# Prepare data
test_passenger_ids = test.PassengerId
test.drop('PassengerId', axis=1, inplace=True)


# Pandas calls these dummies. Let's do it for both sets.
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)


# Features and labels.
features = train.drop('Survived', axis=1)
labels = train.Survived
n_components = len(features.columns)


# Build network
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adamax
from keras.constraints import maxnorm
def create_model(learn_rate=0.001, beta_1=0.9, beta_2=0.999, momentum=0, dropout_rate=0.0, weight_constraint=0):
	model = Sequential()
	model.add(Dense(n_components, activation='relu', input_shape=(n_components,)))#, kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(n_components*2, activation='relu'))
	model.add(Dense(n_components*3, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	# model.compile(optimizer=SGD(lr=learn_rate, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])
	model.compile(optimizer=Adamax(lr=learn_rate, beta_1=beta_1, beta_2=beta_2), loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# Refine model.
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
model_wrap = KerasClassifier(build_fn=create_model)
params = {
  'validation_split': [0.2],
  'epochs': [50],
  #'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
  'learn_rate': np.linspace(0.001, 1, 50), #[0.001, 0.01, 0.1, 0.2, 0.3],
	'beta_1': np.linspace(0.8, 1.0, 10),
	'beta_2': np.linspace(0.8, 1.0, 10),
  # 'momentum': np.linspace(0.0, 1.0, 10), #[0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
  'dropout_rate': np.linspace(0.0, 1.0, 10),#[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  'weight_constraint': [0, 1, 2, 3, 4, 5]
}
cv = RandomizedSearchCV(
	estimator=model_wrap,
	param_distributions=params,
	cv=3,
	n_iter=50,
	n_jobs=-1
)
# cv_results = cv.fit(features, labels)
# print('Score: {} / {}'.format(cv.best_score_, cv.best_params_))
# best_model = cv.best_estimator_.model
# import matplotlib.pyplot as plt
# plt.plot(best_model.history.history['loss'], label='loss')
# plt.plot(best_model.history.history['val_loss'], label='val loss')
# plt.legend()
# plt.show()

import sys
# sys.exit()

# Score: 0.8170594841275285 / {'weight_constraint': 3, 'validation_split': 0.2, 'learn_rate': 0.2, 'epochs': 80, 'dropout_rate': 0.3}
# Score: 0.8136924807605251 / {'weight_constraint': 1, 'validation_split': 0.2, 'learn_rate': 0.1, 'epochs': 80, 'dropout_rate': 0.3}
# Score: 0.8170594841275285 / {'weight_constraint': 3, 'validation_split': 0.2, 'momentum': 0.1111111111111111, 'learn_rate': 0.6326567346938775, 'epochs': 50, 'dropout_rate': 0.1111111111111111}
# Score: 0.7968574639924046 / {'weight_constraint': 2, 'validation_split': 0.2, 'learn_rate': 0.08255102040816327, 'epochs': 50, 'dropout_rate': 1.0, 'beta_2': 0.8444444444444444, 'beta_1': 0.8444444444444444}

from keras.callbacks import EarlyStopping
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
]

from keras.utils import to_categorical
# model = create_model(0.6326567346938775, 0.1111111111111111, 0.1111111111111111, 3)
model = create_model(learn_rate=0.02138, beta_1=0.999, beta_2=1.0, dropout_rate=0.3)
history = model.fit(features, to_categorical(labels), validation_split=0.25, epochs=300, callbacks=callbacks)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

#### Predict & Submit
predicted_survival = model.predict_classes(test)
print(predicted_survival)

submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': predicted_survival
})
submission.to_csv('./submissions/deep.csv', index=False)
print('Done!')