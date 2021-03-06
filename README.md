# Optical & SAR Fusion for Building Footprint Extraction using a U-Net Architecture 

** This is not an official entry into the spacenet 6 challenge - nor does it follow the guidelines of the challenge ** 

To those not familiar, the spacenet6 dataset includes Synthetic Aperture Radar (Capella Space) and Optical Multispectral satellite imagery (Maxar's Worldview-2). 
The dataset was released publicly to drive one of the many spacenet challenges. Spacenet 6 in particular was a challenge centred around building footprint extraction using artificial intelligence - by using a fusion of optical and SAR data. I did not enter this challenge and thus explored the data with my own goals in mind. The main purpose was to write some tools around geospatial preprocessing, as well as U-Net training in tensorflow. I would not consider the model results below 'robust' , as I did not perform hyperparameter optimization nor did I strive to improve the results at all. This is really just a benchmark run. Any requests to update/modify/maintain the code will respectfully be ignored.

More on the challenge here: https://www.cosmiqworks.org/archived-projects/spacenet-6/

Dataset location (Amazon AWS): https://registry.opendata.aws/spacenet/


![Alt text](/figures/Detection_Rate_Plot-zoomed.png?raw=true "test")
![Alt text](/figures/F1_Scores.png?raw=true "test")
![Alt text](/figures/Figure_1.jpg?raw=true "test")
![Alt text](/figures/Figure_2.jpg?raw=true "test")
![Alt text](/figures/Figure_3.jpg?raw=true "test")
![Alt text](/figures/Figure_4.jpg?raw=true "test")
![Alt text](/figures/Figure_5.jpg?raw=true "test")
![Alt text](/figures/Figure_6.jpg?raw=true "test")
