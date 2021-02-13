# LowResImageRecognision
This system was designed to investigate different approaches that can boost the performance of a low resolution dataset. The approaches used will be fine-to-coarse knowledge transfer, multitask learning and auxiliary learning. The dissertation can be read [here](reports/dissertation.pdf)

# Datasets
To link the dataset to the system, ensure that the config file is correctly configured

The boxcars dataset can be found at https://github.com/JakubSochor/BoxCars

The stanford cars dataset can be found at https://ai.stanford.edu/~jkrause/cars/car_dataset.html

To create the down sampled version of stanford, run the `jupyter_files/DownsampleImages.py`. Ensure the path is correct in the file
# Training
The `training.py` script is used to train the network. There are a number of flags that can be added to train a network. The required flags are listed below
```
python3 train.py –model-id 33 –model-version 2 –dataset-version 2 –adam
```

# Testing
To test the dataset, use the commands below. Ensure to specify the correct flags values
```
python3 test.py –model-id 34 –model-version 2 –dataset-version 2
```
