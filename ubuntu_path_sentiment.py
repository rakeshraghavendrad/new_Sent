#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import gc
import speech_recognition as sr
import warnings
warnings.filterwarnings('ignore')
from monkeylearn import MonkeyLearn
from moviepy.editor import ffmpeg_tools  
import moviepy.editor as mp
import pandas as pd
import numpy as np
import boto3
from boto3.session import Session
import sys
import time 
import language_tool_python
from textblob import TextBlob
import librosa
import requests
import os, shutil
import json
#--------#
path = '/home/ubuntu/new_Sent/videos/'
path1 = "/home/ubuntu/new_Sent/aud/"
get_File_names = 'https://ibridge360.com/api/UserAnswersAndScores/getFilenames'
populate_DB = 'https://ibridge360.com/api/UserAnswersAndScores/addSentimentData'
#------------#
# Video's Downloded from S3
url = get_File_names 
#url = 'http://dev.ibridge360.com/api/UserAnswersAndScores/addSentimentData'
re = requests.get(url)
#r = re.content
js = json.loads(re.text)
#--------------#
ACCESS_KEY = str(input('Please enter Access Key:-     '))
SECRET_KEY =str(input('Please enter Secret Access Key:-    '))
session = Session(aws_access_key_id=ACCESS_KEY,aws_secret_access_key=SECRET_KEY)
s3 = session.resource('s3')
your_bucket = s3.Bucket('ibridge') 
filename = []
for s3_file in your_bucket.objects.all():
    g = s3_file.size/1000000 
    if(g > 3.00):
        filename.append(s3_file.key)       
s3 = boto3.client('s3',aws_access_key_id = ACCESS_KEY,aws_secret_access_key= SECRET_KEY)
filename1 = []
#--------------#
for s2_file in filename:
    if(s2_file.startswith("videos/")):
        if any(s2_file[7:] in s for s in js):
            filename1.append(s2_file)
            #print('file exist')
        #else:
            #filename1.append(s2_file)
for l in range(len(filename1)):
    path2 = "/home/ubuntu/new_Sent/"
    s3.download_file('ibridge',filename1[l],path2+filename1[l])
print("videos are downloded ")
# DATA Analysis Part
global a    
for filename in os.listdir(path):
    if filename.endswith(".mp4"):
        b = os.path.getsize(path+filename)
        b = b/1000000
        if (b >2.00):
            gc.collect()
            file_name = path+filename
            wav_file_name = path1+filename +".wav"
            ffmpeg_tools.ffmpeg_extract_audio(file_name, wav_file_name)
            du = librosa.get_duration(filename = wav_file_name)
            if(du < 60):
                du = du,'in Sec'
            else:
                du = du/60
                du = round(du, 2)
                du = du, 'in Min'
             
            r = sr.Recognizer()
            with sr.AudioFile(wav_file_name) as source :
                     audio = r.record(source)
            try:
                a = r.recognize_google(audio)
                #print(a)
                if (len(a) > 0):
                    gc.collect()
                    ml = MonkeyLearn('2d14ec2555abc7f9e2aefee982e212d9091e310a')
                    data = [a]
                    model_id = 'cl_pi3C7JiL'
                    result = ml.classifiers.classify(model_id, data)
                    t = result.body[0]
                    classifications = t['classifications']
                    x = pd.DataFrame(classifications)
                    X1 = x.drop(['tag_id'], axis = 1)
                    X2 = X1.drop(['tag_name'], axis = 1)
                    X2.columns =[""]
                    X2 = X2.values
                    X2 = X2.ravel()
                    X3 = X1.drop(['confidence'], axis = 1)
                    X3.columns =[""]
                    X3 = X3.values
                    X3 = X3.ravel()
                    X2 = X2 * 100
                    list1 = [X3 , ':-' ,str(X2),'%']
                    str1 = ''.join(map(str, list1))
                    tl = language_tool_python.LanguageTool('en-IN')
                    txt = a
                    m = tl.check(txt)
                    c = (len(m)-1)
                    def convert(lst):
                        return ([i for item in lst for i in item.split()])
                    data= [a]
                    lst =  data
                    lst = convert(lst)
                    mistakes = 0
                    for x in lst:
                        a1 = TextBlob(x)
                        if (a1.correct() != x):
                            mistakes = mistakes + 1

                    l1 =[filename]
                    l2 =[str1]
                    l3 =[str(c)]
                    l4 =[mistakes]
                    l5 = [du]
                    #data = pd.DataFrame(list(zip(l1, l2, l3, l4, l5)))
                    data = pd.DataFrame(list(zip(l1,l2, l3, l4, l5)))
                    data.columns =['Filename','Sentiment Analysis','Total Grammatical Errors','Total Spelling Errors','Total duration']
                    data.to_csv("Analysis_AWS_data.csv",mode='a')
                    print("Data transcribed Successful")
                    del a
                   
            except:
                print("Sorry, I did not get that") 
                error = "Issues while recording/microphone muted/disturbed environment"
                l1 =[filename]
                l4 =[error]
                data_e = pd.DataFrame(list(zip(l1, l4)))
                data_e.columns =['Filename', 'ERROR MESSAGE']
                data_e.to_csv("Error_Analysis.csv",mode='a')
                print("Data transcribed to error")

if os.path.isfile('Error_Analysis.csv'):
#try:
    df = pd.read_csv('Analysis_AWS_data.csv')
    du1=df.drop_duplicates()
    df1 = du1.loc[:, ~du1.columns.str.contains('^Unnamed')]
    df1.index.name = 'Index'
    #df1 =df1.drop(df1.index[1])
    #blankIndex=[''] * len(df1)
    #df1.index=blankIndex
    #df1.to_csv("Final_Analysis.csv",mode='a')
    print('Analysis Done')
#----------------#
    ds = pd.read_csv("Error_Analysis.csv")
    ds1=ds.drop_duplicates()
    ds1 = ds1.loc[:, ~ds1.columns.str.contains('^Unnamed')]
    ds1.index.name = 'Index'

    #blankIndex=[''] * len(ds1)
    #ds1.index=blankIndex

#----------------#
    fl = pd.concat([df1,ds1])
    fl.to_csv("fnl_n.csv")
#n_df = pd.read_csv("fnl.csv")
#co = pd.concat([n_df,f1])
#co.to_csv("fnl.csv")
    print('final .csv is printed go check it out from try block')
#-------------------#
    url1 = populate_DB
    s = pd.read_csv('fnl_n.csv')
    s.fillna('N/A', inplace = True)
    for index, row in s.iterrows():
         myobj = {
            "communicationVideoName":str(row['Filename']),
            "finalAssessment":str(row['Sentiment Analysis']),
            "totalDuration":str(row['Total duration']),
            "spellingErrors":str(row['Total Spelling Errors']),
            "grammaticalErrors":str(row['Total Grammatical Errors']),
            "errorMessage":str(row['ERROR MESSAGE'])
            }

         x = requests.post(url1, data = myobj)

         print(x.text)
         print("I am From try block")
#----------------#
    folder = path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    import os, shutil
    folder1 = path1
    for filename in os.listdir(folder1):
        file_path = os.path.join(folder1, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
#------------------#
    file = 'Analysis_AWS_data.csv'
    file1 = 'Error_Analysis.csv'
    if(os.path.exists(file1) and os.path.isfile(file1)):
            os.remove(file1)
            print("file deleted",file1)
    if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)
            print("file deleted",file)
#---------------------------#
    file2 = 'fnl_n.csv'
    if(os.path.exists(file2) and os.path.isfile(file2)):
              os.remove(file2)
              print("file deleted",file2)
              print("program completed successfully and Populated values in databases")
#--------------------------------------#
#except:
else:
    df = pd.read_csv('Analysis_AWS_data.csv')
    du1=df.drop_duplicates()
    df1 = du1.loc[:, ~du1.columns.str.contains('^Unnamed')]
    df1.index.name = 'Index'
    #df1 =df1.drop(df1.index[1])
    #blankIndex=[''] * len(df1)
    #df1.index=blankIndex
    #df1.to_csv("Final_Analysis.csv",mode='a')
    print('Analysis Done')
#----------------#
    folder = path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    import os, shutil
    folder1 = path1
    for filename in os.listdir(folder1):
        file_path = os.path.join(folder1, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
#----------------------------------#
    url = populate_DB
    s = pd.read_csv('Analysis_AWS_data.csv')
    s.fillna('N/A', inplace = True)
    for index, row in s.iterrows():
         myobj = {
            "communicationVideoName":str(row['Filename']),
            "finalAssessment":str(row['Sentiment Analysis']),
            "totalDuration":str(row['Total duration']),
            "spellingErrors":str(row['Total Spelling Errors']),
            "grammaticalErrors":str(row['Total Grammatical Errors']),
            "errorMessage":str('N/A')
            }

         x = requests.post(url, data = myobj)

         print(x.text)
         print("I am From Except block")
#------------------------------------#
    file2 = 'Analysis_AWS_data.csv'
    if(os.path.exists(file2) and os.path.isfile(file2)):
            os.remove(file2)
            print("file deleted",file2)
            print("program completed successfully and Populated values in databases")
#----------------------------------------------------------------------------------------#

