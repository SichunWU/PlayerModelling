import numpy as np
import seaborn as sns
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import json
import dataProcessing


def MLP_model(X,y):
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


def scatter_plot(X, y):
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    # X_test = X_test.to_numpy()
    # y_test = y_test.to_numpy()

    X = X.to_numpy()
    y = y.to_numpy()
    model = keras.models.load_model("data/MLP_model.h5")
    #model.summary()
    y_predicted = [model.predict(input_data.reshape(1, -1)) for input_data in X]

    # scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_predicted, c='b', label='Predicted Difficulty', alpha=0.3)
    plt.plot([1, 5], [1, 5], 'r--', label='Ideal Line')  # 添加理想的线性关系，x=y
    plt.title('Rating vs Predicted')
    plt.xlabel('Rating')
    plt.ylabel('Predicted')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/MLP_model_scatter_plot.png')
    plt.show()
    plt.close()

def heatmap_plot(X, y):
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    df = pd.concat([X, y], axis=1)
    correlation_matrix = df.corr()

    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('data/MLP_model_heatmap_plot.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    data_path = 'data/output.csv'
    input_data = pd.read_csv(data_path)
    X, y = dataProcessing.create_dataset(input_data)
    # MLP_model(X, y)

    history = 'data/MLP_model_train.json'
    # top_result(history)
    # plotting(history)

    # scatter_plot(X, y)
    heatmap_plot(X, y)
