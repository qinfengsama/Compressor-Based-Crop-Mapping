# Compressor-Based-Crop-Mapping
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

