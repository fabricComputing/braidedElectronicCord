# import pandas as pd
# import csv
# import numpy as np
#
# add2 = "./data_set/Wear-resistant/train/data2_extract/"
# add3 = "./data_set/Wear-resistant/train/"
# add4 = "wear-resistant"
# length = 50
#
# def get_data_train(f, extract_length):
#     data_train = [[]] * length * 8
#     for i in range(8):
#         for n in range(length):
#             data_2 = pd.read_csv(add2 + f[i] + "/" + f[i] + str(
#                 n + 1) + ".csv")  # 读取csv文件,names=['dCh1','dCh2','dCh3','dCh4']
#             data_2 = np.array(data_2)
#             data_2 = data_2.astype(np.float32)
#             # print(data_2)
#             # data_train[i*96+n] = data_2[2]
#             for j in range(4):
#                 # print(data_train[i * 50 + n])
#                 print(data_train[-1])
#                 print(data_2[j+1][0])
#                 if len(data_2) != 0:
#                     data_train[i * length + n] = np.insert(data_train[i * length + n], 0, data_2[j + 1])
#             # data_train[i * 96 + n]=np.append(data_train[i * 96 + n], np.array(i, dtype=int))
#             data_train[i * length + n] = np.insert(data_train[i * length + n], 0, i)
#             print(add2 + f[i] + "/" + f[i] + str(n + 1) + ".csv")
#             print(len(data_train[0]))
#             # print(data_train)
#     l = list(range(extract_length * 4))
#     column_title = ['type'] + l
#     with open(add3+add4 + ".csv", 'w', newline='') as f_c:
#         csv.writer(f_c).writerow(column_title)
#         for i in data_train:
#             csv.writer(f_c).writerow(i)
#
#     # calculate_range("ca", 0, 10, 0.05, extract_length)
#     # calculate_range("hd", 1, 10, 0.05, extract_length)
#     # calculate_range("l", 2, 10, 0.02, extract_length)
#     # calculate_range("m", 3, 10, 0.02, extract_length)
#     # calculate_range("nd", 4, 10, 0.05, extract_length)
#     # calculate_range("sj", 5, 10, 0.05, extract_length)
#     # calculate_range("u", 6, 10, 0.05, extract_length)
#     # calculate_range("zw", 7, 10, 0.05, extract_length)
#
# if __name__ == '__main__':
#     f = ["ca", "hd", "l", "m", "nd", "sj", "h", "zw"]
#     extract_length = 130
#     get_data_train(f, extract_length)

import pandas as pd
import csv
import numpy as np

add2 = r"D:\EPIC\Code\Python\capacitance\capacitance\waveclassification\data_set\h_test\data2_extract\\"
add3 = r"D:\EPIC\Code\Python\capacitance\capacitance\waveclassification\data_set\h_test\\"
add4 = "test"
length = 100

def get_data_train(f, extract_length):
    data_train = [[]] * length * 3
    for i in range(3):
        for n in range(length):
            data_2 = pd.read_csv(add2 + f[i] + "/" + f[i] + str(
                n + 1) + ".csv")  # 读取csv文件,names=['dCh1','dCh2','dCh3','dCh4']
            data_2 = np.array(data_2)
            data_2 = data_2.astype(np.float32)
            # print(data_2)
            # data_train[i*96+n] = data_2[2]
            for j in range(4):
                # print(data_train[i * 50 + n])
                print(data_train[-1])
                print(data_2[j+1][0])
                if len(data_2) != 0:
                    data_train[i * length + n] = np.insert(data_train[i * length + n], 0, data_2[j + 1])
            # data_train[i * 96 + n]=np.append(data_train[i * 96 + n], np.array(i, dtype=int))
            data_train[i * length + n] = np.insert(data_train[i * length + n], 0, i)
            print(add2 + f[i] + "/" + f[i] + str(n + 1) + ".csv")
            print(len(data_train[0]))
            # print(data_train)
    l = list(range(extract_length * 4))
    column_title = ['type'] + l
    with open(add3+add4 + ".csv", 'w', newline='') as f_c:
        csv.writer(f_c).writerow(column_title)
        for i in data_train:
            csv.writer(f_c).writerow(i)

    # calculate_range("ca", 0, 10, 0.05, extract_length)
    # calculate_range("hd", 1, 10, 0.05, extract_length)
    # calculate_range("l", 2, 10, 0.02, extract_length)
    # calculate_range("m", 3, 10, 0.02, extract_length)
    # calculate_range("nd", 4, 10, 0.05, extract_length)
    # calculate_range("sj", 5, 10, 0.05, extract_length)
    # calculate_range("u", 6, 10, 0.05, extract_length)
    # calculate_range("zw", 7, 10, 0.05, extract_length)

if __name__ == '__main__':
    f = ["h", "hh", "hl"]
    extract_length = 130
    get_data_train(f, extract_length)
