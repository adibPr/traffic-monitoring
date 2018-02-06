# traffic-monitoring
Traffic monitoring from video using python and opencv

Mostly based on thesis of Jacub Sochor that you can read in [Here](https://medusa.fit.vutbr.cz/traffic/data/papers/2014-JakubSochor-Thesis.pdf).
To use it you have to create folder data, and put ground truth obtained from [Here](https://github.com/JakubSochor/BrnoCompSpeed) and create sample data and download some sample from data set link.
Current tree view : 
```
├── background.py
├── config.py
├── data
│   ├── gt
│   │   └── 2016-ITS-BrnoCompSpeed
│   └── sample
│       ├── session0_center.avi
│       ├── session0_left.avi
│       └── session0_right.avi
├── iterator.py
├── main.py
└── README.md
```


## Data
I use sample of [BrnoCompSpeed](https://medusa.fit.vutbr.cz/traffic/research-topics/traffic-camera-calibration/brnocompspeed/) dataset of session 1. They already included the calibration result that I used in here.

## TO-DO
- [x] download data
- [x] resampling so each view start at same time (using frame ID)
- [x] background subtraction
- [x] load VP
- [ ] automatically find common plane
- [ ] perspective inverse mapping
- [ ] combine multicamera
- [x] draw bounding box
- [ ] draw 3D bounding box
