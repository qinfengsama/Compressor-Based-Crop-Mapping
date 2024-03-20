# Compressor-Based-Crop-Mapping
## Dataset
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
--k 2
~~~

## Acknowledgement
We borrowed the code of deep learning models used in the comparative experiment from [BreizhCrops](https://github.com/dl4sits/BreizhCrops), 

