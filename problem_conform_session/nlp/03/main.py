import pickle
import os
from janome.tokenizer import Tokenizer
import pandas as pd

t = Tokenizer()

# 形態素解析関数の再定義
def tokenize(text):
    tokens = t.tokenize(text)
    return [token.base_form for token in tokens if token.part_of_speech.split(',')[0] in ['名詞', '動詞']]

# モデルのロード
model_filename = 'text_classification_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

test = pd.read_csv("test.csv")

new_predictions = loaded_model.predict(test['text'])
# print(new_predictions)

submission_df = pd.DataFrame({
    'id': test['id'],
    'label': new_predictions
})

# Save the DataFrame to a CSV file
submission_df.to_csv("submission.csv", index=False)