#testapi
from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
#Global scaler for preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0:5]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 4])
    return np.array(dataX), np.array(dataY)

def preprocess_and_create_datasets(df, look_back):
    df2 = df[['Price per unit_x', "Volume_x","Price per unit_y", "Volume_y", 'Sales']]
    data_df = df2 
    data_scaled = scaler.fit_transform(data_df[['Price per unit_x', 'Volume_x', 'Price per unit_y', 'Volume_y', 'Sales']])
    testX, testY = create_dataset(data_scaled, look_back)
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))
    return testX, testY


def duplicate_column(array):
    duplicated_array = np.repeat(array, 5, axis=1)
    return duplicated_array

def fifth_func(resulting_array):
    fifth_column = resulting_array[:, 4]
    reshaped_fifth_column = fifth_column.reshape(-1, 1)
    return reshaped_fifth_column

def convert_to_n_x_5(data):
    n = len(data) // 5
    new_array = np.zeros((n, 5))
    for i in range(n):
        new_array[i, 0] = data[i * 5]
        new_array[i, 1] = data[i * 5 + 1]
        new_array[i, 2] = data[i * 5 + 2]
        new_array[i, 3] = data[i * 5 + 3]
        new_array[i, 4] = data[i * 5 + 4]
    return new_array

def generate_date_dataframe(start_date):
    start_date = pd.to_datetime(start_date)
    dates = pd.date_range(start=start_date, periods=30, freq='D')
    df = pd.DataFrame(dates, columns=["Date"])
    return df






# model entry
@app.route('/modelA/predict', methods=['POST'])
def predict():
    new_data = request.json

# Convert the JSON data into a DataFrame
    new_data_df = pd.DataFrame.from_dict(new_data)

#try:
    model_path = "model1A.h5"
    model = tf.keras.models.load_model(model_path)
    look_back = 10 
    testX, testY = preprocess_and_create_datasets(new_data_df, look_back)
    predictions = model.predict(testX)
    pred1 = duplicate_column(predictions)
    pred2=scaler.inverse_transform(pred1)
    prediction = fifth_func(pred2)
    testY5=convert_to_n_x_5(testY)
    TestY=scaler.inverse_transform(testY5)
    testy5=fifth_func(TestY)
    # END
    # now the date answer format
    # Example usage
#df4 = pd.DataFrame(date1, columns=['Date'])
    df7= pd.DataFrame(prediction, columns=['Prediction'])
    df8= pd.DataFrame(testy5, columns=['Actual_Value'])
    #Answer_df = pd.concat([df7,df8], ignore_index=True)
    Answer_df= df7


    
    #END
    # Format the dictionary as per the required structure
    
    # Return the formatted data as JSON
    
    return jsonify(Answer_df.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='0.0.0.0', debug=True)


