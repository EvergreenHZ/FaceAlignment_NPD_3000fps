#NOTE 
Most of the code comes from [here](...)
# Train
Since we have prepare the data, `$ ./FaceAlignment prepare` will generate two text file `train.txt` and `test.txt` under `../data/68`. These text files are used for training and testing, each line points out an image path and face bounding box in this image with facial points location. We use VJ detector provided by OpenCV, you can use your own face detector to generate these two text file.

`$ ./FaceAlignment train` will start training and result will be lied in the directory `model`. `$ ./FaceAlignment test` will test the model on test data. If you have a Desktop Environment on the Linux, `$ ./FaceAlignment run` will do the prediction over test data by presenting a window to show the result.
