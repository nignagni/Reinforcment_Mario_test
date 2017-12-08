# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:36:37 2017

@author: Nick
"""
from __future__ import division
import numpy as np
import cv2
import time
import pyautogui 
from directkeys import PressKey, ReleaseKey, W, A, S, D, Z, X,F1
from grabscreen import grab_screen
from getkeys import key_check
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt

#Load training game over and game win don
file_name =   'training_data.npy' 
game_over = np.load('game_over.npy')
game_winner = np.load('game_winer.npy')


tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
a = game_over.reshape([1,len(game_over.flatten())])
inputs1 = tf.placeholder(shape=[1,len(game_over.flatten())],dtype=tf.float32)
Watt = tf.Variable(tf.random_uniform([len(game_over.flatten()),4],0,0.01))
Qout = tf.matmul(inputs1,Watt)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
updateModel = trainer.minimize(loss)



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



if os.path.isfile(file_name):
    print('File existins, loading previous data!')
    training_data = list(np.load(file_name))
    
else:
    print('file does exist, starteing fresh')
    training_data = []

#again this processing will not be used...
def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 3)
    except:
        pass



#ROI not used, but could be useful later
def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    #processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    #For edges
    #processed_img = cv2.GaussianBlur(processed_img, (3,3), 0 )
    
    #Region of interest, will not be used... YET... muahaha
    #vertices = np.array([[10,500],[10,300], [300,200], [500,200], [800,300], [800,500]], np.int32)
    #processed_img = roi(processed_img, [vertices])
    
        #                       edges, again, for use later??
    #lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, 20, 15)
    #draw_lines(processed_img,lines)
    
    return processed_img

def main():
    
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    screen = np.zeros([120,200,3])
    i = 0
    paused = False
    
    while True:
        
        
        screen = grab_screen(region=(80,80,500,500))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (80,60))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        print('Frame took {} seconds'.format(time.time()-last_time))
        
        #opencv recognize 
        if (screen[10:][:] == game_over[10:][:]).all():
            paused = False
            print('Game over man')
            Reset_Mario()
        
        if (screen[30:][:] == game_winner[30:][:]).all():
            paused = False
            print('Winner Winner Chicken Dinner')
            Reset_Mario()
        
        
        
        last_time = time.time()
        
        
        #This is just a random jump 50% of the time and you can pause it
        if not paused:
            if random.randint(1, 100)%2 == 0:
                jump()
            else:
                straight()
            time.sleep(.05)
            
        keys = key_check()
        
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(X)
                ReleaseKey(D)
                time.sleep(1)
        
        #To save the training data obtained.
        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name,training_data)
        
        
        cv2.imshow('window', screen)
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        

init = tf.initialize_all_variables()
# Set learning parameters
y = .99
e = 0.1
num_episodes = 100
#create lists to contain total rewards and steps per episode
jList = []
rList = []

def sample_environment():
    screen = grab_screen(region=(80,80,500,500))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen, (80,60))
    s1 = game_over.reshape([1,len(game_over.flatten())])
    
    return s1,screen

with tf.Session() as sess:
    sess.run(init)
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        Reset_Mario()
        [s,screen1] = sample_environment()
        rAll = 0
        d = False
        j = 0
        print('We ReSET!')
        #The Q-Network
        while j < 500:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s})
            average = np.average(allQ)
            
            print('Just a test', a[0], 'With all of them ',allQ, "Whats the average? ", average)
            
            if np.random.rand(1) < e:
                Pick_action(np.random.randint(4, size=1))
            else:
                Pick_action(a)
            
            time.sleep(.2) 
            #Get new state and reward from environment
            [s1,compare] = sample_environment()
            
            #opencv recognize 
            if (compare[10:][:] == game_over[10:][:]).all():
                paused = False
                print('Game over man')
                r = 0
                d = True
            elif (compare[30:][:] == game_winner[30:][:]).all():
                paused = False
                print('Winner Winner Chicken Dinner')
                r = 100
                d = True
            else:
                r = 0
            
            dist = np.linalg.norm(screen1-compare)/1000
            screen1 = compare
            r = r + dist
            
            print ("reward" ,r)
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:s1})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            
            #print('Q1 ', Q1, 'Maximum Q1 ',maxQ1, "Whats the average? ", average)
            
            targetQ[0,a[0]] = r + y*maxQ1
            
            #print('Updated Target ', targetQ)

            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,Watt],feed_dict={inputs1:s,nextQ:targetQ})
            
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
        
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")



#main()