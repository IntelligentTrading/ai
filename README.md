# ai
A package to systematically train/retrain networks.
Uses artemis package, which allows to keep track of all experiments which were ran, then retrieve their results.


### package description
- train.py, the main file, you can ran it on AWS in background
- /notebooks, examples of notebook
- /data/data_sources - get raw data from DB to form dataset
- /data/datasets - convert raw proces, volumes etc into matrix datasets
- /models - keras (or may be tensorflow later) models

### installation
- create an AWS instance with GPU (p2.xlarge)
- get access to it via ssh:
ssh -i "/Users/alex/PycharmProjects/ITT/cert/alex-deeplearn.pem" ubuntu@server.amazonaws.com
- git clone https://github.com/IntelligentTrading/ittai.git
- git pull


### Running
- add all neccesary training experiments to train.add_all_experiments_variants()
- run train.py in a background:     nohup python train.py &








### ARTEMIS 
folders: ~/artemis/experiments/

http://artemis-ml.readthedocs.io/en/latest/experiments.html


Experiment fields and methods:
https://github.com/QUVA-Lab/artemis/blob/master/artemis/experiments/experiment_record.py

