# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:00:03 2017

@author: Nick
"""

from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
from directkeys import PressKey, ReleaseKey, W, A, S, D, Z, X,F1
from grabscreen import grab_screen
from getkeys import key_check
import cv2
import time


WIDTH = 80
HEIGHT = 60
LR = 1e-3

#Load training game over and game win don
file_name =   'training_data.npy' 
game_over = np.load('game_over.npy')
game_winner = np.load('game_winer.npy')
test = np.array_split(game_over[21:],[55,2],axis = 1)

#Unmodified from github awjuliani

class Qnetwork():
    def __init__(self,h_size):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.scalarInput =  tf.placeholder(shape=[None,21168*4],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84*4,84*4,3])
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d( \
            inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
        
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2,4]))
        self.VW = tf.Variable(xavier_init([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,4,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

#Also directly copied from awjuliani       
class experience_buffer():
    def __init__(self, buffer_size = 5000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])

def processState(states):
    return np.reshape(states,[21168*4])

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
        
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .5 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 5000#How many steps of training to reduce startE to endE.
num_episodes = 3000 #How many episodes of game environment to train network with.
pre_train_steps = 250 #How many steps of random actions before training begins.
countdown = 240 #number of steps before large negative reward is encured!

max_epLength = 250 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/annealing_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)
    

#My stuff to control mario

def keys_to_output(keys):
    #[D,ELSE,DX,Z]
    output = [0,0,0,0]
    
    if 'D' in keys:
        output = [1,0,0,0]
    elif 'D' and 'X' in keys:
        output = [0,0,1,0]
    elif 'Z' in keys:
        output = [0,0,0,1]
    else:
        output = [0,1,0,0]
    
    return output

def straight():
    PressKey(D)
    ReleaseKey(X)
    ReleaseKey(Z)
    ReleaseKey(A)
    

def jump():
    PressKey(X)
    PressKey(D)
    ReleaseKey(Z)
    
def run():
    PressKey(Z)
    
def back():
    ReleaseKey(D)
    PressKey(A)
    

def Reset_Mario():
    PressKey(F1)
    ReleaseKey(F1)

def Pick_action(a):
    if a == 0:
        straight()
    if a == 1:
        jump()
    if a == 2:
        run()
    if a == 3:
        back()

def sample_environment():
    screen = grab_screen(region=(80,80,500,500))
    screen2 = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen, (84,84))
    screen2 = cv2.resize(screen2, (WIDTH,HEIGHT))
    return screen,screen2

#Create a blank "short term" memory buffer.
next = np.zeros((HEIGHT, WIDTH*4,3), np.uint8)




with tf.Session() as sess:
    sess.run(init)
    
    
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        #Create a blank "short term" memory buffer.
        next = np.zeros((HEIGHT, WIDTH*4,3), np.uint8)
        
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation
        Reset_Mario()
        [s,compare] = sample_environment()
        
        nextp = next[:HEIGHT,:WIDTH*3,:3]
        next[:HEIGHT,WIDTH:WIDTH*4,:3] = nextp
        next[:HEIGHT, :WIDTH,:3] = s
        
        s = processState(next)
        
        d = False
        rAll = 0
        j = 0
        countdown_counter = 0
        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            countdown_counter+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,4)
                print('THats so RAnD0m')
            else:
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
            

            print('Just a test', a)
            
            Pick_action(a)
            time.sleep(.2)
            
            #s1,r,d = env.step(a)
            #get states, rewards and quitting
            
            #Get new state and reward from environment
            [s1,compare] = sample_environment()
            nextp = next[:HEIGHT,:WIDTH*3,:3]
            next[:HEIGHT,WIDTH:WIDTH*4,:3] = nextp
            next[:HEIGHT, :WIDTH,:3] = s1
            
            comparenew = np.array_split(compare[21:],[55,2],axis = 1)
            #opencv recognize 
            if (comparenew[0] == test[0]).all():
                paused = False
                print('Game over man')
                r = -30
                d = True
            elif (compare[22:][:] == game_winner[22:][:]).all():
                paused = False
                print('Winner Winner Chicken Dinner')
                r = 1000
                countdown_counter = 0
            else:
                r = 0
            
            s1 = processState(next)
            dist = np.linalg.norm(s-s1)/1000
            
            r = r + dist
            if (r < 3 and r>=0):
                r = 0
            
            if (countdown_counter >= countdown):
                r = -100
            
            print ("reward" ,r, " countdown ",countdown_counter)
            
            
            #s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
            
            if total_steps > pre_train_steps:
                
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    print ("RUNNING TRAIN")
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    #Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    
                    updateTarget(targetOps,sess) #Update the target network toward the primary network.
            rAll += r
            s = s1
            
            if d == True:

                break
        
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        #Periodically save the model. 
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print(total_steps,np.mean(rList[-10:]), e)
    saver.save(sess,path+'/model-'+str(i)+'.ckpt')
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
plt.plot(rList)
plt.plot(jList)
