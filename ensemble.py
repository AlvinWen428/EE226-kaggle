import pandas as pd

file_list = ['end_result.csv', 'end_result (1).csv',
             'wen_result (6).csv', 'wen_result (7).csv', 'wen_result (8).csv']

all_data = []
for file in file_list:
    all_data.append([])
    data_frame = pd.read_csv(file)
    print(data_frame.shape)
    for i in range(data_frame.shape[0]):
        all_data[-1].append(data_frame.iloc[i, 1])

end_result = []
for j in range(len(all_data[0])):
    result2num = {}
    for i in range(len(all_data)):
        if all_data[i][j] not in result2num:
          result2num[all_data[i][j]] = 0
        result2num[all_data[i][j]] += 1
    num2result = {m:k for k,m in result2num.items()}
    end_result.append(num2result[max(num2result.keys())])

ID = [i for i in range(10000)]
out_dict = {'ID':ID, 'Category':end_result}
columns = ['ID', 'Category']
df = pd.DataFrame(out_dict)#构建DataFrame结构
df.to_csv('./end_result (2).csv', index=False, columns=columns)