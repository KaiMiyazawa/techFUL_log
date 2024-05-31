import pickle
import os
import pandas as pd
import numpy as np

# モデルのロード
model_filename = 'text_classification_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

test = pd.read_csv("test.csv")
test.drop(['store_id', 'day'], axis='columns', inplace=True)
if '\ufeffid' in test.columns:
    test = test.drop('\ufeffid', axis=1)
else:
    print("カラム 'id' は存在しません。")
print(test)


test.fillna(method = 'ffill', inplace=True) 

a = list(test.columns)
new_predictions = loaded_model.predict(test[a])
# print(new_predictions)

submission_df = pd.DataFrame({
    'id': test['id'],
    'sales': new_predictions
})

# Save the DataFrame to a CSV file
submission_df.to_csv("submission.csv", index=False)