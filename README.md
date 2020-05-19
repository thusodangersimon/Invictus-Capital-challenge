# Repo for Invictus Capital Challenge

The data for the challenge has been containerized. To run the code, please install docker-2.3.0.2 and make.

To build docker image run the command:

`$make create-all`

That will build the docker image and launch a jupyter lab instance at port 8888.
Copy log in token given in the command line.

In the notebook folder there are notebooks part 1 and 2 which contain code to train the model.

notebooks with the suffix `_run.ipnb` have the saved model ready to run.    

# Troubleshooting
* Sometimes the docker take 2 comands to build, so if the build fails the first time try the make command again.

* Using `$make help` Will show the other commands 

* to stop container use ctrl-c. To launch the container again `$make run-container`
