#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.optimize as spo
import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
import pandas
import datetime


def diff_eqs(theta,t,alpha,beta,gamma,average_follower):
    s = theta[0]
    e = theta[1]
    i = theta[2]
    k = average_follower
    result = np.zeros((3))
    n = s+e+i
    
    #dS(t)/dt
    result[0] = -(s*alpha) -(s*(s/n)*k*alpha) -(e*(s/n)*k*beta) -(i*(s/n)*k*gamma)
    #dE(t)/dt
    result[1] = -(e*beta) +(s*(s/n)*k*alpha) +(e*(s/n)*k*beta) +(i*(s/n)*k*gamma)
    #dI(t)/dt
    result[2] = +(s*alpha) +(e*beta)
    
    return result

def nonlinear_least_square_param(model,xdata,ydata):      
    time_total = xdata
    data_record = ydata
    
    I0 = ydata[0]
    E0 = 0
    S0 = 326*10**6 - I0 - E0
    alpha_init = 0.05
    beta_init= 0.4
    gamma_init = 0.2
    average_follower = 30
    initial_guess=[alpha_init,beta_init,gamma_init,average_follower]
    
    theta=[S0,E0,I0]
    
    #Bounds of the diffision rates
    bnds = ((0,1),(0,1),(0,1),(0,S0))

    popt = spo.minimize(sse(model,theta,time_total,data_record),initial_guess,method="TNC",bounds=bnds).x
    return popt

def sse(model,theta,time_total,data_record):
    def result(x):
        Nt = spi.odeint(model,theta,time_total,args=tuple(x))
        
        difference = data_record - Nt[:,2]
        diff = np.dot(difference,difference)
        return diff
    return result


df = pandas.read_csv("./inauguration.csv", encoding="ISO-8859-1",header=0)

#Sort by date
df = df.sort_values(by='created')

infect_counter = 0
exposure_counter = 0
#x,y data
df_x = []
df_exposed = []
df_infected = []

#Get first x then compute time passed from here on
first_x = datetime.datetime.strptime(df.iloc[0,5],'%Y-%m-%d %H:%M:%S')

#Seconds in day to convert datetime to simple int
seconds_in_day = 60*60*24
average_follower = 223

#Append time passed
for index,row in df.iterrows():
    time_tweeted = row['created']
    duration_seconds = datetime.datetime.strptime(time_tweeted,'%Y-%m-%d %H:%M:%S') - first_x
    df_x.append(duration_seconds.days * seconds_in_day + duration_seconds.seconds)
    
    exposure_counter += average_follower
    infect_counter += 1
    df_exposed.append(exposure_counter)
    df_infected.append(infect_counter)


theta=[326*(10**6),0,0]
#Turn seconds to hours
for i in range(len(df_x)):
    df_x[i] = df_x[i]/3600


param = nonlinear_least_square_param(diff_eqs,df_x,df_infected)
y = spi.odeint(diff_eqs,theta,df_x,args=tuple(param))
 
#y = spi.odeint(diff_eqs,theta,df_x,args=(alpha,beta,gamma,average_follower))
S,E,I = y.T


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd')
ax.plot(df_x, S,'b',label='Susecptible')
ax.plot(df_x,E,'g',label='Exposed')
ax.plot(df_x,I,'r',label='Infected')

ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
legend = ax.legend()


ax.set_xlabel('Time(Hours)')
ax.set_ylabel('Infected')

axes=plt.gca()
axes.set_ylim(0,df_infected[-1]+1000)
plt.show()



    