from sklearn import datasets
from quantile_forest import RandomForestQuantileRegressor

train_data = datasets.load_digits()
digits_features, digits_labels = train_data['data'], train_data['target']
qrf = RandomForestQuantileRegressor()
qrf.fit(digits_features, digits_labels)
y_pred = qrf.predict(digits_features) # Median prediction for each input

# Sample from the random quantile forest then calculate mmd