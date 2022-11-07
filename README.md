# **Imperceptible, Designable, and Scalable** Braided Electronic Cord

## Data acquisition

A multichannel capacitance-analyzing circuit with a Bluetooth module (Zhihe 01RC) was designed and manufactured by Linkzill. The chip used in the device is a thin-film transistor (TFT) semiconductor chip independently developed by Linkzill Technology. The chip can realize high-throughput digital liquid programming control, high-throughput and high-precision array signal control and detection, and large-area and low-cost biological signal acquisition. This chip serves the industrial application of life science, sensor display, and other cross fields Scenes.

## Dataset

We provide part of the interaction dataset for testing and validation in the [raw_mini](https://github.com/fabricComputing/braidedElectronicCord/tree/main/raw_mini) folder.

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

## Core code

### Data preprocessing

In this part, you need to complete the extraction of actions, label them, extract features and generate training data.

### Training the model

1. [train1_LSTM.py](https://github.com/fabricComputing/braidedElectronicCord/blob/main/code/train1_LSTM.py)
2. [train2_ML.py](https://github.com/fabricComputing/braidedElectronicCord/blob/main/code/train2_ML.py)
3. [train3_autoML.py](https://github.com/fabricComputing/braidedElectronicCord/blob/main/code/train3_autoML.py)

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
