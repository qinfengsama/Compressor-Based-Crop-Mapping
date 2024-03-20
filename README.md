# A Pixel Can Be Compressed as a Spectral-Temporal Text
## Citation

## Introduction
The accuracy of crop mapping, based on multi-spectral temporal data has been significantly improved through the use of deep learning. However, traditional deep learning can be computationally intensive, requiring large amounts of data and millions of parameters, which can make it `expensive' to utilize and optimize. Inspired by research on natural language processing, we consider a pixel in satellite images as a 'spectral-temporal' text. Specifically, the proposed symbol representation algorithm is used to convert the band reflectance of all pixels into symbol representations. Then, the Normalized Compression Distance (NCD) between the time series of any two pixels in the same band and the NCD between the spectral sequences at the same timestamp are calculated to obtain the average NCD between any two pixels. Finally, based on it, classification is implemented using simple k-nearest-neighbor classifier (kNN).  Without any trainable parameters, our method achieves results that are competitive with deep learning methods across four sub-scenes of the PASTIS dataset. It even outperforms the average of nine advanced deep learning methods.  Our method also excels in the few-shot setting, where labeled data are too scarce to train neural network effectively. Further experiments also validate the effectiveness of it for early-season classification.

## Dataset
The original dataset originates from [pastis-benchmark](https://github.com/VSainteuf/pastis-benchmark)

Since the dataset is too large, you can download the data we use from the following links of baiduyun:

https://pan.baidu.com/s/1zYGEa1OOLbVkexjC1FAwSQ?pwd=4dbe (access code: 4dbe)

## Usage
### Dependencies

### Ours
As an example, use the following command to run our method on the T30UXV parcel.
~~~
cd model/ours/

python 0227_tsc_gzip.py \
--dataset pastis \
--area t30uxv \
--period 43 \
--compressor gzip \
--concat_mode bp_pb \
--code char \
--alphabet_len 51 \
--mapping equal_interval \
--str_code normal \
--train_num 0.5 \
--k 8
~~~

### Deep learning models
We also provide our pretrained models for inference in [xxx](xxx)
#### TempCNN, LSTM and InceptionTime
~~~
cd model/TempCNN_LSTM_InceptionTime/

python tempcnn.py \
--areas t30uxv \
--train_num 0.5 \
--period 43 \
--model "Choose the model from TempCNN, LSTM, or InceptionTime."
~~~

## Acknowledgement
The deep learning models implementations from this repository are based on the following github repositories.
- TempCNN, LSTM and InceptionTime from [BreizhCrops](https://github.com/dl4sits/BreizhCrops)

