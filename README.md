# carla-imitation-learning
## Note : This repo is still in developpement so it might change in the Future
### Before getting into the imitation learning part we need to setup our environnement for this we use the virtualen extension which coul be installed by executing the following line of code in your Anaconda prompt :
$ pip install virtualenv
now that we installed the virtualenv extension we need to create our python virtual environnement, in our case we setup our environnement on python 3.8 as follow :

$ virtualenv myenv --python=python3.8

The Virtual environnement now created we can activate it by using :

Windows : myenv\Scripts\activate

Linux : source myenv/bin/activate

#### Requirements installation
To install all the dependencies to run our code, please use this following line of code, make sure to be in the provided Requirements.txt file directory :

$ conda install --file Requirements.txt

## How to :

### Please launch the CarlaUE4.exe or CarlaUE4.sh before running any python the data gathering code

### Behavioural cloning :

We can divide this part of the project in three parts, just as in machine learning, so the first part would be the data gathering, next we'd have the model training, and finally loading the model loading into Carla agent in order to control our agent which is the spawned car with the model predictions.

#### Data gathering :

We make our data collection in the data_collection_no_map_data.py file, for the data collection we have 2 choices, we can either collect our data from the built in Carla auto pilot for this please exectute this command in your Anaconda prompt : 

$ python data_collection_no_map_data.py --autopilot --folder desired/data/saving/folder/

and we can also make our data collection with a manual control of the car using the keyboard arrow keys (please make sure to have your keyboard set in english on your operating system for this) by executing the following command in your anaconda prompt :

$ python data_collection_no_map_data.py --folder desired/data/saving/folder/

#### Model training and testing :

for this task we use the Xception, since we are using pytorch instead of tensorflow for Cuda compatibility reasons we used the Xception model converted to the torch format from the following repository : 


https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py 


which should be downloaded and put in your carla folder in order to import it to our python code file.

Now to run the training and testing program, please use the following command :

$ python Train_test_torch.py --file your/path/and/file_name.pt

#### Model loading and prediction for controlling the vehicle  :

That is the easiest part now that everything is setup, in order to use the you just need to write this code below in your Anaconda prompt (Make sure that CarlaUE4.exe is running) :

$ python behavioral_cloning_model_agent.py --file your/path/and/file_name.pt

Once this code executed, wait for the model to be loaded, then the vehicle should start moving on its own.

Congratulations ! You have now executed the Behavioral Cloning program. 

Note : if you want to skip the training part we have provided the test_model_.pt file that you can use directly for the Behavioral Cloning model agent.
























































