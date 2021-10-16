import numpy as np
from flask import jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import flask
from keras.models import load_model
import pickle
from flask import request



app = flask.Flask(__name__)
# from datetime import datetime
model = load_model('manufacturingCost.h5')


with open('manufacturing_hist.pickle', 'rb') as f:
    history = pickle.load(f)



@app.route('/predictManufacturingCost', methods=['POST'])
def predict():

    data = list(request.form.values())[0]
    # request_data = request.get_json()
    print('***************************************')
    # url = request_data['dataset']
    date = 10
    print(data)
    print('***************************************')
    # Importing Training Set
    df = pd.read_csv(data)
    df.info()
    # Check null values
    sum(df.isnull().sum())
    # Check 0 values
    print((df == 0).sum())
    df = df.replace(0, np.NAN)

    max = df['cost'].max()
    print('+++++++++++++++++++++++++++++++++++++++')
    print(max)
    print('+++++++++++++++++++++++++++++++++++++++')
    train_dates = pd.to_datetime(df['date'])

    Y = df[['cost']]

    # Variables for training
    cols = list(df)[1:11]

    df_for_training = df[cols].astype(float)

    # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)

    # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    # In this example, the n_features is 2. We will make timesteps = 3.
    # With this, the resultant n_samples is 5 (as the input data has 9 rows).
    trainX = []
    trainY = []

    n_future = 1  # Number of days we want to predict into the future
    n_past = 45  # Number of past days we want to use to predict the future

    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i - 2:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))

    # Forecasting...
    # Start with the last day in training date and predict future...
    n_future = date  # Redefining n_future to extend prediction dates beyond original n_future dates...
    forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

    forecastTrainng = model.predict(trainX)
    forecastFuture = model.predict(trainX[-n_future:])  # forecast

    # Perform inverse transformation to rescale back to original range
    # Since we used 9 variables for transform, the inverse expects same dimensions
    # Therefore, let us copy our values 9 times and discard them after inverse transform
    forecast_copies = np.repeat(forecastFuture, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

    forecast_Tranning_copies = np.repeat(forecastTrainng, df_for_training.shape[1], axis=-1)
    y_pred_tranning = scaler.inverse_transform(forecast_Tranning_copies)[:, 0]

    actual_cost = np.repeat(trainX, 10, axis=-1)
    y_actual = scaler.inverse_transform(trainX)[:, 0]
    real = y_actual[:, 0]

    df_forecast_score = pd.DataFrame({'Predict': np.array(y_pred_tranning), 'Real': y_actual[:, 0]})
    print(df_forecast_score)

    Real_value = df_forecast_score.Real[40:]
    Pedict_value = df_forecast_score.Predict[40:]

    # checking R2 Score
    from sklearn.metrics import r2_score
    r2 = r2_score(Real_value, Pedict_value)
    print('-------------------------------------')
    print('r2 score for perfect model is', r2)
    print('-------------------------------------')


    # Convert timestamp to date
    forecast_dates = []
    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())

    df_forecast = pd.DataFrame({'date': np.array(forecast_dates), 'cost': y_pred_future})
    print(df_forecast)

    dfList = df_forecast.values.tolist()
    return jsonify(
        name="Manufacturing cost prediction",
        description='Total manufacturing cost including variable',
        prediction=dfList)
    # return jsonify(y_pred_future.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
