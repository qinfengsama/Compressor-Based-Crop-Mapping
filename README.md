# low-resource crop classification from multi-spectral time series using lossless compressors

## Dataset
Since the dataset is too large, you can download the data we use from the following links of google drive:
[Dataset](https://drive.google.com/drive/folders/1eMuwGf54EcDpi8Ed8mXVb0F9FnbxH5up?usp=sharing)

Once you have downloaded the data in `rs.npy` format and the ground truth in `gt.npy`, please place them in the `dataset_name/` path, where `dataset_name` can be `Pastis`, `German` and `France` ('France' refers to the 'T31TFM-1618 dataset' mentioned in the paper).

## Our Method
### Dependencies
~~~
pip install pandas
pip install tqdm
pip install openpyxl
~~~

### Usage
You can reproduce the results of our method for the **Section 3.1 Comparisons with Deep Learning Models**, **Section 3.2 Few-Shot Learning** and **Section 4 Analyses** by executing the following command line:

~~~
python run_ours.py
~~~

Note that the files `icpr_ours.py`, `run_ours.py` and the above three dataset folders must be in the same directory.

### Notes
To improve efficiency, we have used the `ProcessPoolExecutor` class to achieve CPU multiprocessing. If you encounter any errors related to this, you can resolve them by adding the parameter `max_workers` to the line of code `with ProcessPoolExecutor() as executor:` and setting its value to less than the number of CPU cores on your machine.

## Acknowledgement

All the experiment datasets are public, and we obtain them from the following links: [pastis-benchmark](https://github.com/VSainteuf/pastis-benchmark)

