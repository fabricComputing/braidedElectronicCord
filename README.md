# **Imperceptible, Designable, and Scalable** Braided Electronic Cord

## Data acquisition

A multichannel capacitance-analyzing circuit with a Bluetooth module (Zhihe 01RC) was designed and manufactured by Linkzill. The chip used in the device is a thin-film transistor (TFT) semiconductor chip independently developed by Linkzill Technology. The chip can realize high-throughput digital liquid programming control, high-throughput and high-precision array signal control and detection, and large-area and low-cost biological signal acquisition. This chip serves the industrial application of life science, sensor display, and other cross fields Scenes.

## Dataset

We provide the interaction dataset for testing and validation in the [raw_mini](https://github.com/fabricComputing/braidedElectronicCord/tree/main/raw_mini) folder.

There are one hundred sets of data under each interactive command.

| Label | Interactive command |
| ----- | ------------------- |
| ca    | long press          |
| h     | pinch position 1    |
| hd    | swipe               |
| l     | pinch position 2    |
| m     | pinch position 3    |
| nd    | twist               |
| sj    | double click        |
| zw    | grab                |

## Main code

### Data preprocessing

In this part, complete the extraction of actions, label them, and generate training data

1. [get1_start_index.py](https://github.com/fabricComputing/braidedElectronicCord/blob/main/code/get1_start_index.py)
2. [get2_csv_plo.py](https://github.com/fabricComputing/braidedElectronicCord/blob/main/code/get2_csv_plo.py)
3. [get3_data_train.py](https://github.com/fabricComputing/braidedElectronicCord/blob/main/code/get3_data_train.py)

### Model training

4. [train4_LSTM.py](https://github.com/fabricComputing/braidedElectronicCord/blob/main/code/train4_LSTM.py)
5. [train5_ML.py](https://github.com/fabricComputing/braidedElectronicCord/blob/main/code/train5_ML.py)
6. [train6_autoML.py](https://github.com/fabricComputing/braidedElectronicCord/blob/main/code/train6_autoML.py)

### Interactive application implementation 

[MusicPlayer](https://github.com/fabricComputing/braidedElectronicCord/tree/main/code/MusicPlayer)

Functions：

-   WebSocket communication
-   Bluetooth communication
-   Control music interface
  -   Status: Playing, Pausing
  -   Cut Songs: Previous, Next
  -   Volume: increase, decrease
  -   List: Shuffle, Loop Single, Sequential Play
