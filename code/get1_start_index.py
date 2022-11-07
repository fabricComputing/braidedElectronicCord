import pandas as pd
import numpy as np
import csv

add1 = ""  # raw data address
add2 = "./data2_extract/" # destination folder address
length = 50


def calculate_range(f, m, l, min_mean, extracted_length):
    print(f, m, l)
    for n in range(length):
        print(add1 + f + "/" + f + str(n + 1) + ".csv")
        data_dCh = pd.read_csv(add1 + f + "/" + f + str(n + 1) + ".csv", skiprows=[0, 1],
                               usecols=[1, 3, 5, 7])
        print("success")
        data_dCh = np.array(data_dCh).T
        i = data_dCh[1]
        i_start = 0
        i_end = 0
        i_mid = 0
        for j in range(0, len(i)):
            l_window = i[j:j + l]
            l_mean = np.mean(l_window)
            if l_mean > i[0] + min_mean and i_mid == 0:
                i_start = j
                i_mid = 1

        indexs[m].append(i_start)


        index_range[m].append(i_end - i_start)
    with open("./data_set/data1_startIndex/" + f + "_start.csv", 'w', newline='') as f_csv:
        csv.writer(f_csv).writerow(indexs[m])
    f_csv.close()
    for n in range(length):
        data_Ch = pd.read_csv(add1 + f + "/" + f + str(n + 1) + ".csv", skiprows=[0, 1], usecols=[0, 1, 3, 5, 7],
                              dtype={'Ch1': np.float32, 'Ch2': np.float32, 'Ch3': np.float32,
                                     'Ch4': np.float32})  # 读取csv文件,names=['dCh1','dCh2','dCh3','dCh4']
        data_Ch = np.array(data_Ch).T
        with open(add2 + f + "/" + f + str(n + 1) + ".csv", 'w', newline='') as f_c:
            csv.writer(f_c).writerow(i for i in range(extracted_length))
            for i in range(5):
                csv.writer(f_c).writerow(data_Ch[i][indexs[m][n] - 10:indexs[m][n] + extracted_length - 10])
        f_c.close()


if __name__ == '__main__':
    index_range = [], [], [], [], [], [], [], []
    indexs = [[] for i in range(8)]
    indexe = [[]] * 8
    extract_length = 130
    e_length = 20
    offset = 0.05
    calculate_range("ca", 0, e_length, offset, extract_length)
    calculate_range("hd", 1, e_length, offset, extract_length)
    calculate_range("l", 2, e_length, offset, extract_length)
    calculate_range("m", 3, e_length, offset, extract_length)
    calculate_range("nd", 4, e_length, offset, extract_length)
    calculate_range("sj", 5, e_length, offset, extract_length)
    calculate_range("h", 6, e_length, offset, extract_length)
    calculate_range("zw", 7, e_length, offset, extract_length)