# DLG_Pong_DQN_CNN_TensorFlow
# Introduction: 
  This repo is for the SADE 2020 program. The program aim to provides students a hand-on experiences with Artificial Intelligence and Machine Learning. The program has 2 teams for the AI project and each team is asked to provide a technical document which explained the basic concepts of AI and ML to whom have little to no experience with the concepts. Our team aims
# Requirments:
  The project was based on the Reinforcement Learning of a Pong game, implemented by Karpathy. http://karpathy.github.io/2016/05/31/rl/
  Our team is asked to improve the model and document the process as well as explaining the relevant concepts assuming the readers have little to no knowledge of these concepts 
# Accomplishment:
  We implemented a new model based on Karpathy's implementation by using Keras and Tensorflow with Python as well as comparing the learning curved of different types of Gradient Descent algorithm.
  The process was well documented along with relevant Machine Learning concepts in a 100 page technical document https://drive.google.com/file/d/1uCo9uGp3rqo6DvYOHdCNW8RD5pgNVnte/view
 # Files: 
  Each model includes 2 file .py and .ipynb. This allows the model can be run on a local machine or with Google Colab. The instruction is documented in the documentation under section 7.1
    - karpathy_pong_model: Karpathy's implementation of the Pong game with his deep Q neural network model using numpy and the RMSprop gradient descent algorithm (section 7.2)
    - dqn_ping_keras: an improvement in implementation the model using Keras with the same RMSprop gradient descent algorithm (section 7.3)
    - cnn_pong_keras: the model's neural network architecture is changed to Convolutional Neural Network. The gradient descent algorithm is also changed to Adam (section 7.4).
