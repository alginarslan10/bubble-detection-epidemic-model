import scipy.optimize as spo
import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import datetime
import pickle

global_sse = []
alpha_init = 0.1
beta_init = 0.2
gamma_init = 0.8
average_follower = 300

tweet_folder_name = "AAPL"
tweet_folder_path = "./stocknet-dataset-master/tweet/raw/"+tweet_folder_name+"/*"
average_total_user =291*10**6

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


#Nonlinear fitting part is based on the 
def nonlinear_least_square_param(model,xdata,ydata,average_total_user,initial_guess):      
    time_total = xdata
    data_record = ydata
    
    alpha=initial_guess[0]
    beta=initial_guess[1]
    gamma=initial_guess[2]
    
    I0 = ydata[0]
    E0 = 0
    S0 = average_total_user - I0 - E0
    
    theta=[S0,E0,I0]
    
    xmin=[0,0,0,1]
    xmax=[1,1,1,S0]
    
    bnds = [(low,high) for low, high in zip(xmin,xmax)]
    minimizer_kwargs = dict(method="L-BFGS-B",bounds=bnds)
    
    popt = spo.basinhopping(func=sse(model,theta,time_total,data_record),x0=initial_guess,minimizer_kwargs=minimizer_kwargs).x   

    
    #popt = spo.minimize(sse(model,theta,time_total,data_record),initial_guess,bounds=bnds).x
    return popt

def sse(model,theta,time_total,data_record):
    def result(x):
        (Nt,out) = spi.odeint(model,theta,time_total,args=tuple(x),full_output=1)
        difference = data_record - Nt[:,2]
        diff = np.dot(difference,difference)
        global_sse.append(diff)
        return diff
    return result


#Prepration of data
if not os.path.isfile(tweet_folder_name+"_dataframe.csv"):
    #Prep for reading multiple file 
    df = pd.DataFrame()
    pattern = os.path.join(tweet_folder_path)
    files_to_get = glob.glob(pattern)
    
    tweets_per_day = []
    
    #Read json files and append to dataframe
    for file in files_to_get:
        data = pd.read_json(file,lines=True)
        tweets_per_day.append(len(data))
        df = df.append(data,ignore_index=True)
        
    
    
    #Sort by date
    df = df.sort_values(by="created_at")
    df = df.drop_duplicates(subset=['id'])
    df.to_csv(tweet_folder_name+"_dataframe.csv",index=False)
    with open('tweets_per_day_'+tweet_folder_name+".pickle", 'wb') as fp:
        pickle.dump(tweets_per_day, fp)


df = pd.read_csv(tweet_folder_name+"_dataframe.csv")
df = df.sort_values(by='created_at')
with open ('tweets_per_day_'+tweet_folder_name+".pickle", 'rb') as fp:
    tweets_per_day = pickle.load(fp)

infect_counter = 0
exposure_counter = 0

#x,y data
df_x = []
df_infected = []
df_exposed = []

#Get first x then compute time passed from here on
first_x = datetime.datetime.strptime(df.iloc[0,0][0:-6],'%Y-%m-%d %H:%M:%S')

#Seconds in day to convert timestamp to simple int
seconds_in_day = 60*60*24


#Append time passed
for index,row in df.iterrows():
    
    #Add time passed
    time_tweeted = row['created_at'][0:-6]
    duration_seconds = datetime.datetime.strptime(time_tweeted,'%Y-%m-%d %H:%M:%S') - first_x
    df_x.append(duration_seconds.days * seconds_in_day + duration_seconds.seconds)
    
    #Add exposure and infection
    if exposure_counter <= average_total_user - average_follower:
        exposure_counter += average_follower
    
    infect_counter += 1
    df_exposed.append(exposure_counter)
    df_infected.append(infect_counter)
        

theta=[average_total_user,0,0]
#Turn seconds to hours
for i in range(len(df_x)):
    df_x[i] = df_x[i]/(3600*24)
    

initial_guess = [alpha_init,beta_init,gamma_init,average_follower]


param = nonlinear_least_square_param(diff_eqs,df_x,df_infected,average_total_user,initial_guess)
y = spi.odeint(diff_eqs,theta,df_x,args=tuple(param))

S,E,I = y.T


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd')
ax.plot(df_x, S,'b',label='Susecptible')
ax.plot(df_x,E,'g',label='Exposed')
ax.plot(df_x,I,'r',label='Infected')

#ax.annotate('%0.f' % S[-1], xy=(1,S[-1]), xytext=(8,0), xycoords=('axes fraction', 'data'), textcoords='offset points')
ax.annotate('%0.f' % I[-1], xy=(1,I[-1]), xytext=(8,0), xycoords=('axes fraction', 'data'), textcoords='offset points')
ax.annotate('%0.f' % E[-1], xy=(1,E[-1]), xytext=(8,0), xycoords=('axes fraction', 'data'), textcoords='offset points')


ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
legend = ax.legend()

ax.set_xlabel('Time(Days)')
ax.set_ylabel('Infected(Millions)')

axes=plt.gca()
plt.show()
plt.savefig(tweet_folder_name+".png")