import re
import pandas as pd
from sklearn.preprocessing import StandardScaler

data_path = 'data/1021.csv'
input_path = 'data/cleaned.csv'

# Unify data format
def clean_data(raw_data):
    # concatenate items in a row and replace |
    raw_data['concatenated'] = raw_data.apply(lambda row: '@'.join(map(str, row)), axis=1)
    raw_data['concatenated'] = raw_data['concatenated'].apply(replace_patterns)

    # split it again and update form
    raw_data['concatenated'] = raw_data['concatenated'].str.split('@')
    raw_data['playerID'] = raw_data['concatenated'].apply(lambda x: x[0])
    raw_data['reportTimeStamp'] = raw_data['concatenated'].apply(lambda x: x[1])
    raw_data['reportDateTime'] = raw_data['concatenated'].apply(lambda x: x[2])
    raw_data['sessionID'] = raw_data['concatenated'].apply(lambda x: x[3])
    raw_data['levelName'] = raw_data['concatenated'].apply(lambda x: x[4])
    raw_data['winnerID'] = raw_data['concatenated'].apply(lambda x: x[5])
    raw_data['winnerHP'] = raw_data['concatenated'].apply(lambda x: x[6])
    raw_data['framesCount'] = raw_data['concatenated'].apply(lambda x: x[7])
    raw_data['pauseframes'] = raw_data['concatenated'].apply(lambda x: x[8])
    raw_data['levelTime'] = raw_data['concatenated'].apply(lambda x: x[9])
    raw_data['pcTravelDistance'] = raw_data['concatenated'].apply(lambda x: x[10])
    raw_data['inputTimes'] = raw_data['concatenated'].apply(lambda x: x[11])
    raw_data['bulletsFiredPC'] = raw_data['concatenated'].apply(lambda x: x[12])
    raw_data['pcRating'] = raw_data['concatenated'].apply(lambda x: x[13])
    raw_data = raw_data.drop(columns=['Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'concatenated'])

    raw_data.to_csv('data/cleaned.csv', index=False)

# Define and replace pattern
def replace_patterns(raw_data):
    patterns = [
        (r'\|(\d+-\d+)@(\d+)\|', r'\1.\2'),
        (r'\|(\d+)@(\d+)\|', r'\1.\2'),
        (r'\|(\d+)@(\d+)\|', r'\1.\2'),
    ]

    for pattern, replacement in patterns:
        raw_data = re.sub(pattern, replacement, raw_data)

    return raw_data

# Create data for training
def create_data(raw_data):
    # 1.Number of deaths in the current level (if dead, including current death)
    raw_data.loc[raw_data['winnerID'] == 1, 'winnerID'] = 0
    raw_data.loc[raw_data['winnerID'] != 0, 'winnerID'] = 1
    raw_data = raw_data.rename(columns={'winnerID': 'deathNum'})
    # cumulative sum
    raw_data['deathNum'] = raw_data.groupby(['sessionID', 'levelName'])['deathNum'].cumsum()

    # 2.Time of current run (either cleared or died)
    raw_data['levelTime'] = raw_data.groupby(['sessionID', 'levelName'])['levelTime'].cumsum()

    # 3.Total distance traveled in the level
    raw_data['pcTravelDistance'] = raw_data.groupby(['sessionID', 'levelName'])['pcTravelDistance'].cumsum()

    # 4.Total active time in the current session
    raw_data['pauseframes'] = raw_data.groupby(['sessionID', 'levelName'])['pauseframes'].cumsum()
    raw_data['framesCount'] = raw_data.groupby(['sessionID', 'levelName'])['framesCount'].cumsum()
    raw_data['activeTime'] = raw_data['levelTime'] * (1 - (raw_data['pauseframes']/ raw_data['framesCount']))

    # 5.Input frequency in keys per minute of the current level
    raw_data['inputTimes'] = raw_data.groupby(['sessionID', 'levelName'])['inputTimes'].cumsum()
    raw_data['inputFreq'] = raw_data['inputTimes'] / raw_data['activeTime']

    raw_data.to_csv('data/output.csv', index=False)

# Create dataset
def create_dataset(raw_data):
    scaler = StandardScaler()

    raw_data[['deathNum', 'levelTime', 'pcTravelDistance', 'activeTime', 'inputFreq']] = scaler.fit_transform(
        raw_data[['deathNum', 'levelTime', 'pcTravelDistance', 'activeTime', 'inputFreq']])

    X = raw_data[['deathNum', 'levelTime', 'pcTravelDistance', 'activeTime', 'inputFreq']]
    y = raw_data['pcRating']

    return X, y

if __name__ == '__main__':
    raw_data = pd.read_csv(data_path)
    clean_data(raw_data)
    cleaned_data = pd.read_csv(input_path)
    create_data(cleaned_data)


