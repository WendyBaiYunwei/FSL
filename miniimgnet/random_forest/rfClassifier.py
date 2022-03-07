from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import pickle

# test RFL

print('loading dataset')
with open('rf_trainX.pkl', 'rb') as f:
    trainX = pickle.load(f)

with open('rf_trainY.pkl', 'rb') as f:
    trainY = pickle.load(f)

print(trainX.shape, trainY.shape)
print('start training')
classifier = RandomForestRegressor(n_estimators = 200, random_state = 42, max_features=4)
classifier.fit(trainX, trainY)
del trainX
del trainY

with open('rf_testX.pkl', 'rb') as f:
    testX = pickle.load(f)

with open('rf_testY.pkl', 'rb') as f:
    testY = pickle.load(f)
print(testX.shape, testY.shape)
# load testX, testY
print('start testing')
preds = classifier.predict(testX)
accuracy = accuracy_score(preds, testY)
print('Accuracy:', accuracy*100, '%.')