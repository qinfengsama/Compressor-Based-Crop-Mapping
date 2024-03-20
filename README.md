# Compressor-Based-Crop-Mapping
## dataset
Due to the dataset being too large, you can download the data we use from the following links of baiduyun:

https://pan.baidu.com/s/1zYGEa1OOLbVkexjC1FAwSQ?pwd=4dbe (access code: 4dbe)

## Usage
### ours
~~~
cd utils/src
python 0227_tsc_gzip.py \
--dataset 'pastis' \
--area 't30uxv' \
--period '43' \
--compressor 'gzip' \
--concat_mode 'bp_pb' \
--code 'char' \
--alphabet_len '51' \
--mapping 'equal_interval' \
--str_code 'normal' \
--train_num '0.5' \
--k '2'
~~~
