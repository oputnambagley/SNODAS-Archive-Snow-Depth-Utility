**SNODAS Archive Snow Depth Utility**

## ---

## This utility allows users to easily interface with current and historical snow depth information by retrieving geographic snow depth data from the [National Snow and Ice Data Center's Snow Data Assimilation System archives](https://noaadata.apps.nsidc.org/NOAA/G02158/).

## Description

---

The goal of this application is to effectively query and process snow depth data from the NSIDC SNODAS archive and display it in a user friendly, digestible format. The utility does this by providing geographic and time-depth visualizations of snow depth data from the contiguous United States and southern Canada since 2009. 
This utility has 3 main functions:

1. Plotting snow depth maps of certain regions at a specified time and displaying the deepest detected snow depths in said region  
2. Plotting snow depth progression at a certain location over a specified date range  
3. Querying snow depth at a certain location at a specified time

## Getting Started

---

### Dependencies

This program requires Python 3 to be installed along with the following libraries:

* **numpy**  
* **matplotlib**  
* **skimage**  
* **cartopy**  
* **requests**  
* **geopy**

These libraries can be installed with the following commands:

```shell
pip install numpy
pip install matplotlib
pip install scikit-image
pip install cartopy
pip install requests
pip install geopy
```

### 

### Executing program

To use this program, please download the Snow\_Depth\_Utility.py file along with the county shape files so maps can be properly plotted with all borders. Once downloaded, run the python script within your current working directory following the prompts given. Options will be given in the form of “\[letter/number\]: option” where the input should be the letter or number corresponding to your choice. The program will prompt you until it receives proper input. Zipped files queried from the archive are stored locally in a subdirectory created by the script titled ‘local\_data’. This is so that if the user wants to plot more data from a certain date or date range, the script can access the data much faster without having to redownload it. To clean this locally stored data, use the ‘wipe local data’ option prompted to the user in the starting menu. 

Example Outputs:

Snow depth map example output:![Depth Map][images/Depth_Map.png]  
![Peak Depths Output][Peak_Depths_Output.png]  
Snow depth progression example output:  
![Progression Plot][images/Progression_Plot.png]
Snow depth at point and time example output:
![Depth Query][images/Depth_Query.png]  
Author  
---

Orlando Putnam-Bagley  
[oputnambagley@bates.edu](mailto:oputnambagley@bates.edu)