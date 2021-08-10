from tensorflow.keras.models import load_model
from tensorflow import expand_dims
import numpy as np


def tokenize_text_2d(raw_merchant, max_len=200, emb_dim=8):
    if len(raw_merchant) > max_len:
        raw_merchant = raw_merchant[:max_len]

    str_array = np.zeros((max_len, emb_dim), dtype=np.int32)

    for index, char in enumerate(raw_merchant):
        str_binary = format(ord(char), 'b').zfill(emb_dim)
        str_array[index] = [int(x) for x in str_binary]

    return str_array


def load_and_use_model(model_dir, test_data):
    model = load_model(model_dir)

    X_test_ = np.array([tokenize_text_2d(x) for x in test_data])
    X_test_ = expand_dims(X_test_, axis=-1)
    result = model.predict(X_test_)

    l_extracted = []
    l_open = []
    l_close = []

    for i, text in enumerate(test_data):
        extracted = text[max(round(result[i][0]), 0): round(result[i][1])]
        l_extracted.append(extracted)
        l_open.append(result[i][0])
        l_close.append(result[i][1])

    return l_extracted[0], l_open[0], l_close[0]