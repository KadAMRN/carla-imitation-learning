# carla-imitation-learning
## Note : This repo is still in developpement so it might change in the Future
### Before getting into the imitation learning part we need to setup our environnement for this we use the virtualenv extension which coul be installed by executing the following line of code in your Anaconda prompt :
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

<img width="1280" alt="image" src="https://github.com/KadAMRN/carla-imitation-learning/assets/87100217/a8037d0c-8f20-433e-b53e-97d546a2bc86">


#### Model training and testing :

for this task we use the Xception, since we are using pytorch instead of tensorflow for Cuda compatibility reasons we used the Xception model converted to the torch format from the following repository : 


https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py 


which should be downloaded and put in your carla folder in order to import it to our python code file.

Now to run the training and testing program, please use the following command :

$ python Train_test_torch.py --file your/path/and/file_name.pt

<img width="438" alt="image" src="https://github.com/KadAMRN/carla-imitation-learning/assets/87100217/85cf9127-b3dc-4d33-97f9-5efc52d81493">

![E acc big data](https://github.com/KadAMRN/carla-imitation-learning/assets/87100217/1264329e-625b-4d95-8c68-bcb343886cba)




#### Model loading and prediction for controlling the vehicle  :

That is the easiest part now that everything is setup, in order to use the you just need to write this code below in your Anaconda prompt (Make sure that CarlaUE4.exe is running) :

$ python behavioral_cloning_model_agent.py --file your/path/and/file_name.pt

Once this code executed, wait for the model to be loaded, then the vehicle should start moving on its own.

Congratulations ! You have now executed the Behavioral Cloning program. 

<img width="1278" alt="image" src="https://github.com/KadAMRN/carla-imitation-learning/assets/87100217/22e41eab-6d49-4598-86b5-1faef449fb0b">

Note : if you want to skip the training part we have provided the test_model_.pt file that you can use directly for the Behavioral Cloning model agent in the following link https://we.tl/t-PBDBd1AHiz


### Learning by cheating :

Please refer to the official Learning by Cheating repository on this link https://github.com/dotchen/LearningByCheating for a detailed tutorial on how to run the program (since model files are 102MB and can't be uploaded on this repository).

<img width="408" alt="lbc demo" src="https://github.com/KadAMRN/carla-imitation-learning/assets/87100217/53867830-7b4a-438f-9c7c-0c664e001592">

@inproceedings{chen2019lbc,
  author    = {Chen, Dian and Zhou, Brady and Koltun, Vladlen and Kr\"ahenb\"uhl, Philipp},
  title     = {Learning by Cheating},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2019
},
}

### Dagger Algorithm :
For this algorithm we didn't use Carla simulator for the experience but we chose a less complex environnement which is a gym environnement called "Cartpole", all the details are available on the project_dagger.ipynb


























































