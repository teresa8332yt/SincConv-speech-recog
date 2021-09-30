# SincConv-speech-recognition

## Speech Recognition with SincNet Convolutional Filters
### Introduction
With the rise of the pandemic COVID-19, we designed a machine learning model of speech recognition elevator control system in Mandarin, that could prevent contact infection through buttons. With the goal to apply machine learning on edge devices like elevators, it should have low power consumption. Hence, we used SincNet convolutional filter as the first layer of our model, an efficient way to reduce parameters and simplify computation. With the method of transfer learning, we combined big English Google speech dataset with our small self-collected data in Mandarin to achieve high accuracy. Finally, we simplified the model by implementing quantization and pruning to achieve our goal of low computation.
