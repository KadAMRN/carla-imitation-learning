# carla-imitation-learning
## Note : This repo is still in developpement so it might change in the Future
### Before getting into the imitation learning part we need to setup our environnement for this we use the virtualen extension which coul be installed by executing the following line of code in your Anaconda prompt :
$ pip install virtualenv
now that we installed the virtualenv extension we need to create our python virtual environnement, in our case we setup our environnement on python 3.8 as follow :

$ virtualenv myenv --python=pyton3.8

The Virtual environnement now created we can activate it by using :

Windows : myenv\Scripts\activate

Linux : source myenv/bin/activate

#### Requirements installation
To install all the dependencies to run our code, please use this following line of code, make sure to be in the provided Requirements.txt file directory :

$ conda install --file Requirements.txt

## How to :
### Behavioural cloning :

We can divide this part of the project in three parts, just as in machine learning, so the first part would be the data gathering, next we'd have the model training, and finally loading the model loading into Carla agent in order to control our agent which is the spawned car with the model predictions.

#### Data gathering :

We make our data collection in the data_collection_no_map_data.py file, for the data collection we have 2 choices, we can either collect our data from the built in Carla auto pilot for this please exectute this command in your Anaconda prompt : 

$ python data_collection_no_map_data.py --autopilot --folder desired/data/saving/folder/

and we can also make our data collection with a manual control of the car using the keyboard arrow keys (please make sure to have your keyboard set in english on your operating system for this) by executing the following command in your anaconda prompt :

$ python data_collection_no_map_data.py --folder desired/data/saving/folder/

#### Model training :

for this task we use the Xception, since we are using pytorch instead of tensorflow for Cuda compatibility reasons 






















































