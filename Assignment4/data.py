import pandas as pd
import json

# read json
with open('train_file.json') as f:
    data = json.load(f)
    # to pandas
    df = pd.DataFrame(data)


