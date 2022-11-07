import pandas as pd
import csv
import numpy as np

add2 = "./data2_extract/" # folder address which generated in step 1
add3 = "./data_set/Wear-resistant/train/" # The folder address where the training dataset is generated
add4 = "wear-resistant" # training dataset name
length = 50 # length of training set of each action

def get_data_train(f, extract_length):
    data_train = [[]] * length * 8
    for i in range(8):
        for n in range(length):
            data_2 = pd.read_csv(add2 + f[i] + "/" + f[i] + str(
                n + 1) + ".csv")  # 读取csv文件,names=['dCh1','dCh2','dCh3','dCh4']
            data_2 = np.array(data_2)
            data_2 = data_2.astype(np.float32)
            for j in range(4):
                # print(data_train[i * 50 + n])
                print(data_train[-1])
                print(data_2[j+1][0])
                if len(data_2) != 0:
                    data_train[i * length + n] = np.insert(data_train[i * length + n], 0, data_2[j + 1])
            data_train[i * length + n] = np.insert(data_train[i * length + n], 0, i)
            print(add2 + f[i] + "/" + f[i] + str(n + 1) + ".csv")
            print(len(data_train[0]))
    l = list(range(extract_length * 4))
    column_title = ['type'] + l
    with open(add3+add4 + ".csv", 'w', newline='') as f_c:
        csv.writer(f_c).writerow(column_title)
        for i in data_train:
            csv.writer(f_c).writerow(i)

if __name__ == '__main__':
    f = ["ca", "hd", "l", "m", "nd", "sj", "h", "zw"]
    extract_length = 130
    get_data_train(f, extract_length)