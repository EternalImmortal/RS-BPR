import pandas as pd
import numpy as np

def get_index(path):
    train_df = pd.read_csv(path, delimiter=':', names=['user', 'sequence'])
    train_index = train_df['user'].tolist()
    # print(len(train_index))
    return train_index


index = get_index('data-zf/train_data.txt')[:5]
print(index)

full_output = pd.read_csv('output/output_full.csv', dtype=str).values
print(full_output)
# selected_output = full_output[index]

# print(selected_output)
#
# f = open('output/output.csv', 'w')
# for index, row in selected_output.iterrows():
#     print(row.values.tolist())
#     f.write('%s\n' % row.values)




