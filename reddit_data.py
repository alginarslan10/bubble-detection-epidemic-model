#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas
import os
from Levenshtein import distance as levdist
import datetime

#Absolute path
file_abs_path = os.path.abspath("reddit_wsb.csv/reddit_wsb.csv")


related_words =["gme","gamestop","game"]

author_id_set = {}
author_id_set = set(author_id_set)
number_of_posts = 0
df_x = []

#Read data
df = pandas.read_csv(file_abs_path)

#Sort data
df = df.sort_values(by="timestamp") 


first_x = datetime.datetime.strptime(df._get_value(0,"timestamp"),'%Y-%m-%d %H:%M:%S')
seconds_in_day = 60*60*24

#Iterate dataframef
for index,row in df.iterrows():
    found_flag = 0
    
    
    #Make low case
    title = str(row.title)
    body = str(row.body)
    
    title = title.lower()
    body = body.lower()
    
    title = title.split(" ")
    body = body.split(" ")
    
    time_posted = row["timestamp"]
    duration_seconds = datetime.datetime.strptime(time_posted,'%Y-%m-%d %H:%M:%S') - first_x

    for word in title:
        for related_word in related_words:
            if levdist(word,related_word) <= 1:
                number_of_posts += 1
                author_id_set.add(row.id)
                found_flag = 1
                df_x.append(duration_seconds.days * seconds_in_day + duration_seconds.seconds)
                
                
                
                
                
    #Check if found in title
    if found_flag == 1:
        continue
    
    #Check if in body
    for word in body:
        for related_word in related_words:
            if levdist(word,related_word) <= 1:
                number_of_posts += 1
                author_id_set.add(row.id)
                found_flag = 1
                df_x.append(duration_seconds.days * seconds_in_day + duration_seconds.seconds)
        
        if found_flag == 1:
            break
    
            