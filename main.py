import numpy as np
import seaborn as sns
from tensorflow import keras
from keras import layers, regularizers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import json
import dataProcessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import joblib


def mlp_model(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    model = keras.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(units=64, activation='relu'),
        layers.Dense(units=32, activation='relu'),
        layers.Dense(units=16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    train_history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32,
                              callbacks=[early_stopping])

    model.save('./data/MLP_model.h5')
    with open('./data/MLP_model_train.json', 'w') as f:
        json.dump(train_history.history, f)

    test_history = model.evaluate(X_test, y_test)
    with open('./data/MLP_model_test.json', 'w') as f:
        json.dump({"Loss": test_history[0], "MAE": test_history[1]}, f)


# plotting for a single model
def plotting(filename):
    with open(filename, 'r') as f:
        history = json.load(f)

    # Plot the training/validation loss per epoch
    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename[:-5]+'_loss.png')
    plt.show()
    plt.close()

    # Plot the training/validation MAE per epoch
    plt.plot(history['mae'], label='Training mae')
    plt.plot(history['val_mae'], label='Validation mae')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(filename[:-5] + '_mae.png')
    plt.show()
    plt.close()


# top loss and MAE
def top_result(filename):
    with open(filename, 'r') as f:
        history = json.load(f)

    top_train_loss = min(history['loss'])
    top_val_loss = min(history['val_loss'])
    top_train_mae = min(history['mae'])
    top_val_mae = min(history['val_mae'])
    print('Top-1 training loss:', top_train_loss,
          '\nTop-1 validation loss:', top_val_loss,
          '\nTop-1 training mae:', top_train_mae,
          '\nTop-1 validation mae:', top_val_mae)


def predict(X, y, model_type):
    if model_type == 1:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        model = keras.models.load_model("data/MLP_model.h5")
        model.summary()
    elif model_type == 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = joblib.load('data/RandomForest_model.pkl')

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    y_predicted = [model.predict(input_data.reshape(1, -1)) for input_data in X_test]
    # y_predicted = [np.round(value) for value in y_predicted]
    return y_test,y_predicted


# scatter plot
def scatter_plot(X, y, model_type):
    y_test, y_predicted = predict(X, y,model_type)
    # scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_predicted, c='b', label='Predicted rating', alpha=0.3)
    plt.plot([1, 5], [1, 5], 'r--', label='Ideal Line')
    plt.title('Actual rating vs Predicted rating')
    plt.xlabel('Actual rating')
    plt.ylabel('Predicted rating')
    plt.legend()
    plt.grid(True)
    if model_type == 1:
        plt.savefig('data/MLP_model_scatter_plot.png')
    elif model_type == 2:
        plt.savefig('data/RandomForest_model_scatter_plot.png')
    plt.show()
    plt.close()


# This one measures the linear correlation, it shouldn't be used
def heatmap_plot(X, y, model_type):
    if model_type == 1:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    elif model_type == 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    df = pd.concat([X_test, y_test], axis=1)
    correlation_matrix = df.corr()
    plt.figure(figsize=(15, 10), tight_layout=True)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

    plt.title('Correlation Heatmap')
    if model_type == 1:
        plt.savefig('data/MLP_model_scatter_plot.png')
    elif model_type == 2:
        plt.savefig('data/RandomForest_model_heatmap_plot.png')
    plt.show()
    plt.close()


# error distribution
def error_distribution_plot(X, y, model_type):
    y_test, y_predicted = predict(X, y, model_type)
    if model_type == 1:
        y_predicted = [value[0][0] for value in y_predicted]
    elif model_type == 2:
        y_predicted = [value[0] for value in y_predicted]
    residuals = y_test - y_predicted

    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='blue', bins=10, label='Residuals')
    plt.title('Error Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    # estimate parameters of a normal distribution curve
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_residual, std_residual)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
    plt.legend()
    if model_type == 1:
        plt.savefig('data/MLP_model_error_dist_plot.png')
    elif model_type == 2:
        plt.savefig('data/RandomForest_model_error_dist_plot.png')
    plt.show()
    plt.close()


def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train.values, y_train.values)
    joblib.dump(model, 'data/RandomForest_model.pkl')

    y_pred = model.predict(X_test.values)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)


if __name__ == '__main__':
    data_path = 'data/output.csv'
    input_data = pd.read_csv(data_path)
    X, y = dataProcessing.create_dataset(input_data)
    # mlp_model(X, y)
    # random_forest(X, y)

    history = 'data/MLP_model_train.json'
    # top_result(history)
    # plotting(history)

    # scatter_plot(X, y, model_type=1)
    # scatter_plot(X, y, model_type=2)
    # heatmap_plot(X, y, model_type=1)
    # heatmap_plot(X, y, model_type=2)
    # error_distribution_plot(X, y, model_type=1)
    # error_distribution_plot(X, y, model_type=2)
