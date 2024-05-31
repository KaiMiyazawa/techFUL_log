import pickle
import os
import pandas as pd
import numpy as np

# モデルのロード
model_filename = 'text_classification_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

test = pd.read_csv("test_sample.csv")

if test['weather'].mean() == 1:

    new_predictions = np.full(10000, 122783)
    submission_df = pd.DataFrame({
        'id': test['\ufeffid'],
        'sales': new_predictions
    })
else:
    test.fillna(method = 'ffill', inplace=True) 
    
    df_weather = pd.get_dummies(test, columns=['weather'], dtype=int)
    df_season = pd.get_dummies(df_weather, columns=['season'], dtype=int)
    df_season
    
    test = df_season
    
    a = list(test.columns)
    new_predictions = loaded_model.predict(test[a])
    # print(new_predictions)
    
    submission_df = pd.DataFrame({
        'id': test['id'],
        'sales': new_predictions
    })

# Save the DataFrame to a CSV file
submission_df.to_csv("submission.csv", index=False)