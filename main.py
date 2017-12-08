# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:36:37 2017

@author: Nick
"""

import numpy as np
import cv2
import time
import pyautogui 
from directkeys import PressKey, ReleaseKey, W, A, S, D, Z, X,F1
from grabscreen import grab_screen
from getkeys import key_check
import os
import random



#reset function for mario ... 




def keys_to_output(keys):
    #[D,ELSE,DX]
    output = [0,0,0]
    
    if 'D' in keys:
        output = [1,0,0]
    if 'D' and 'X' in keys:
        output = [0,0,1]
    else:
        output = [0,1,0]
    
    return output

def straight():
    PressKey(D)
    ReleaseKey(X)

def jump():
    PressKey(X)
    PressKey(D)

def Reset_Mario():
    PressKey(F1)
    ReleaseKey(F1)


file_name =   'training_data.npy' 
game_over = np.load('game_over.npy')
game_winner = np.load('game_winer.npy')

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
        
main()