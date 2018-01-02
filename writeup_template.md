# **Behavioral Cloning**


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/steering_hist.png "Steering Hist"
[image2]: ./img/img_7.png "Img1"
[image3]: ./img/img_1234.png "Img2"
[image4]: ./img/img_flipped_7.png "Flipped Img1"
[image5]: ./img/img_flipped_1234.png "Flipped Img2"
[image6]: ./img/img_101.png "Normal Image"


 I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Required Files and Code Quality

#### 1. Submitted Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* video.mp4, a recording of two laps completed by my trained model
* writeup_report.md summarizing the results

#### 2. Functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Code readability

The model.py file contains the code for training and saving the convolution neural network. The file shows the neural network that I
trained, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

While the telemetry in the simulator records various race parameters, as well as simultaneous pictures from three different angles, it turns out that the steering
angle alone and center camera images are sufficient to navigate successfully around Track 1. The problem to be solved is a regression
problem.

#### 1. Model architecture

After experimenting with different models, I settled on the neural
network inspired by the NVIDIA paper. The model consists of five convolutional layers and four flat layers,
with a Dropout layer interposed. Each convolutional layer uses a ReLU
activation function, and the first three use a 2 x 2 stride.

The data is normalized at the beginning, and images are cropped in
order to help the model focus on the relevant details.

#### 2. Attempts to reduce overfitting

The model contains a dropout layer in order to reduce overfitting.
This was needed because the race track contains large stretches of
straight road, which makes the vast majority of captured images contain
a steering angle of zero (per histogram below). This creates a situation of "imbalanced
learning" (similar to the Traffic Sign Classification problem), and
the Dropout layer helps "de-emphasize" some of the zeros.

![alt text][image1]

#### 3. Model parameter tuning

I used the Adam optimizer to train the model. [Adam](https://arxiv.org/pdf/1412.6980.pdf) is an adaptive
training method, where the learning rate is set by a combination of
momentum and RMS prop strategies.


#### 4. Appropriate training data

Driving correctly while using the keyboard requires some good amount
of practice. Because of that, the dataset provided by Udacity is a solid
starting point. It appears to have been recorded while driving from the keyboard (judging by the abrupt changes in steering angles observed
when assembling the data into a movie), but still it contains many laps of correct driving images.

I used a joystick to collect additional training data, focusing on
smoothing out the trajectories at turns. This improved the performance
of the early networks that I have tried, however I found that my final
network performed equally well on just the original data.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from
a network that was known to perform well and tailor it to the problem
at hand.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in a 80/20
ratio.

I normalized each image by using a Lambda keras layer, and I cropped each
image by excluding the top 70 and bottom 25 pixels. This helped the model
focus on the essential information in the picture.

After watching the lecture videos, I suspected that the
[NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)  model
would be the best starting point. But given that in Project 2 I was
able to classify traffic sign images with a variation of LeNet, I
wanted to give that a try. It did not perform well, with the car
crashing immediately after start.

I then tried the model similar to NVIDIA's which is suggested in the lecture videos. It consists of a preprocessing layer, followed by five
convolutional layers, three fully connected layers followed by the output
layer. This model
performed much better, being able to drive up to the first turn with no incidents, when being trained over 10 epochs.

The training data has an inherent bias in the steering angle due to driving around in circles. To combat this, I augmented the data by flipping each image horizontally. This improved the performance to the
point where the car was able to drive up to the end of the bridge.

The curves at the end of the bridge proved to be a major obstacle, due to the sharp turns. The model was not able to hit the required high steering
angles because they were "outliers" based on the data the model had seen.
To address this, I first wrote a pipeline that randomly eliminated
a chunk of the training data with zero angles. This produced the unwanted effect of having the car hit the side of the bridge and getting stuck there at times.

The next thing I tried was introducing a dropout layer right after the
first fully connected layer, and trying some fairly aggressive dropout
probabilities. This worked really well. I experimented with a
range of probabilities between .5 and .95, and found that values
of .75 and .8 performed best, with the car being able to pass the first
curve after the bridge. I noticed that the MSE seemed to be attaining a
minimum at around 8 epochs, so I stopped the training at that point. The
resulting model was able to navigate smoothly and safely around the track.

At the end of the process, the model is able to drive autonomously around the track without leaving the road. The video.mp4 shows two full laps.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes


| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB image   				    |
| Lambda                | normalization x -> x/255 - .5             |
| Cropping 2D           | discard top 70 and bottom 25 pixels       |
| Convolution 5x5     	| 24 filters, 2x2 stride, valid padding,  	|
| RELU 				    |         						                 |
| Convolution 5x5     	| 36 filters, 2x2 stride, valid padding	|
| RELU 				    |         						                 |
| Convolution 5x5     	| 48 filters, 2x2 stride, valid padding	|
| RELU 				    |         						                 |
| Convolution 3x3     	| 64 filters, 1x1 stride, valid padding	|
| RELU 				    |         						                 |
| Convolution 3x3     	| 64 filters, 1x1 stride, valid padding 	|
| RELU 				    |         						                 |
| Fully connected		| 100 nodes				         |
| Dropout                | keep_prob = 0.75                              |
| Fully connected		| Input 100, output 50					|
| Fully connected       | Input 50, output 10       |
| Output				| Input 10, output 1			|		|



#### 3. Training Set & Training Process

To capture good driving behavior, I used the images provided, as
recorded by the center camera. Here is an example image of such image:

![alt text][image6]

To augment the data set and I flipped the images horizontally (while reversing the angles) thinking that this would also eliminate the bias in the steering angle. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image4]

and another


![alt text][image3]
![alt text][image5]


After data augmentation, I had 16,720 data points. I then preprocessed this data by normalizing to a pixel value between 0 and 1
in each channel. I also cropped out the top 75 and bottom 25 pixels.


I randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs for the model I chose was 8, as evidenced by the minimum observed in the MSE numbers.

I used the Adam optimizer so that manually training the learning rate wasn't necessary. I trained the network on a GPU instance on AWS.

### Simulation

The car is able to navigate once around the track, without any tires leaving the drivable surface. A recorded video of the drive is included
in the submission.
