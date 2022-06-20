# ETH Pedestrian Dataset Evaluation of DistNav

This repo contains code for evaluating the DistNav crowd navigation algorithm on the ETH pedestrian dataset. 

More information of the algorithm and evaluation protocol can be found in our paper: [http://www.roboticsproceedings.org/rss17/p053.pdf](http://www.roboticsproceedings.org/rss17/p053.pdf).

To cite our work, please use the following Bibtex:
```
@INPROCEEDINGS{SunM-RSS-21, 
    AUTHOR    = {Muchen Sun AND Francesca Baldini AND Peter Trautman AND Todd Murphey}, 
    TITLE     = {{Move Beyond Trajectories: Distribution Space Coupling for Crowd Navigation}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2021}, 
    ADDRESS   = {Virtual}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2021.XVII.053} 
} 
```

A separate toolbox is available at [https://github.com/MurpheyLab/DistNav](https://github.com/MurpheyLab/DistNav) if you want to test DistNav in other environments or integrate with other algorithms.

## Dataset

The original dataset can be found at: [https://icu.ee.ethz.ch/research/datsets.html](https://icu.ee.ethz.ch/research/datsets.html). The ETH pedestrian dataset is a subset of the BIWI Walking Pedestrians dataset.

We preprocessed the dataset into separate test trials, more details for the data preprocessing can be found in our paper.

## How To Use

To run the evaluation, you only need run the script `evaluate_distnav_eth.py`. By running this script, a `results` directory will be created and test results will saved under this directory. For each test run, there will be a txt file containing all the test information. The data format of the txt file is (each column represents a metric): frame number, number of essential agents, DistNav optimization time, total run time, minimal distance between the removed human agent with other pedestrians, minimal distance between the robot and other pedestrians, path length of the removed human agent, path length of the robot, crowd density.

If you enable `plotting` in the script `evaluate_distnav_eth.py`, you will see the visualization of each test frame, and the plotting will be saved under the same directory in `results`.

We also provide a script to analyze the test results `analyze_results.py`.

## Questions

This code is create by Muchen Sun ([muchen@u.northwestern.edu](muchen@u.northwestern.edu)), who developed the DistNav implementation, and Pete Trautman ([peter.trautman@gmail.com](peter.trautman@gmail.com)), who developed the benchmark and Gaussian processes (GP) configuration pipeline. Please feel free to contact them if you have any question.

