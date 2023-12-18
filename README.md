# 3DRDNN
Project for my master thesis. In this project, we aim to provide DNN model for liver lesion segmentation and develop a 3D version of RECIST metric. Moreover we plan to develop export script for labels to be used in Fiji program. In this way we can show and ask radiologists to evaluate our results. As the results we plan create the new public dataset.

## Plan

- [x] Train / eval / test pipeline
- [x] comparision of Recist 2D and 3D adaptation
- [x] export script to Fiji

## Results

We developed 3D U-Net based model for segmention on 3D patches. This model achieved f1 score ~67%, which is not that good. There is still space to improvment. Main problem is postprocessing. But to say - This model detects very well small changes that 2D models most likely skips.

After graduation, plot, tables and others will be listed here.

## how to use it
TBA


## Data

We used LITS challenge data as main dataset:

[LITS challenge](https://competitions.codalab.org/competitions/17094)

[link to paper](https://arxiv.org/pdf/1901.04056.pdf)