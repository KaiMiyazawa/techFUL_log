import pickle
import os
import pandas as pd

# モデルのロード
model_filename = 'text_classification_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

test = pd.read_csv("test.csv")

a = ['age', 'height', 'TVCM', 'SNS', 'sports', 'history']
new_predictions = loaded_model.predict(test[a])
# print(new_predictions)

submission_df = pd.DataFrame({
    'id': test['id'],
    'purchase': new_predictions
})

# Save the DataFrame to a CSV file
submission_df.to_csv("submission.csv", index=False)