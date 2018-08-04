# ittai
A package to systematically train/retrain networks.
Uses artemis package, which allows to keep track of all experiments which were ran, then retrieve their results.


### installation
- create an AWS instance with GPU (p2.xlarge)
- get access to it via ssh:
ssh -i "/Users/alex/PycharmProjects/ITT/cert/alex-deeplearn.pem" ubuntu@server.amazonaws.com
- git pull (cite command)


### Running
- run train.py in a background:     nohup python train.py &


### package description
- train.py, the main file, you can ran it on AWS in background
- /notebooks, examples of notebook
- /data/data_sources - get raw data from DB to form dataset
- /data/datasets - convert raw proces, volumes etc into matrix datasets
- /models - keras (or may be tensorflow later) models


### description of LSTM models trained here
I suppose to train three models, for 60 / 240 /1440 horizons