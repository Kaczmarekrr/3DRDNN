# 3DRDNN
Project for my master in which we try to translate RECIST metric to 3D and use DNN for automatic segmentation on CT liver images 


## Plan

- Build pipeline for 3D dnn segmation
    - data loader
    - network architecture
    - training
    - results comparision
- Test new ideas:
    - Films as data
    - Data scheduler
    - Weighted loss for films and CT data 

## how to use it
Run notebook pipeline


## Data

### Non-CT datasets: (how many films and data shape)
- VOS (what inside?)
- DAVIS-2017

### CT datasets: (how many scans and data shape)
- LITS_Challenge
- Medical Decathlon (task 3 and 8)
- IRCAD dataset