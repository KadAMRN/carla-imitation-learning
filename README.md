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


for this task we use the Xception, since we are using pytorch instead of tensorflow for Cuda compatibility reasons 






















































