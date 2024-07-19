import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt1
import matplotlib.gridspec as gridspec  # unequal plots
from scipy.optimize import leastsq
from pylab import *

from matplotlib.widgets import Cursor

import os
import tkinter as tk
from tkinter import filedialog
import sys
import json

import builtins


from scipy.optimize import curve_fit
from numpy import arange


work_dir = ''

work_file =''

work_file = filedialog.askopenfilename(initialdir=work_dir, filetypes=[('CSV file', '*.csv')], title='Open CSV file')    

""" Read the curve CSV file.
"""
file_string = work_file
# reads the input file 

# if data file is in an excel file, use the following line: 
# master = pd.read_excel(file_string).to_numpy()

# if using a '.csv' file, use the following line:
data_set = pd.read_csv(file_string).to_numpy()

#
x = data_set[:,0]
max_x=np.max(x)
min_x=np.min(x)
# defines the independent variable. 
#
y =  data_set[:,1]

# --- No changes below this line are necessary ---
# Plotting
fig = plt.figure()
ax = fig.subplots()
#sets the axis labels and parameters.
ax.tick_params(direction = 'in', pad = 15)
ax.set_xlabel('Temperature (K)', fontsize = 15)
ax.set_ylabel('Intensity  (a.u.)', fontsize = 15)
ax.plot(x,y,'bo')
#ax.scatter(x, y, s=15, color='blue', label='Data')
ax.grid()
# Defining the cursor
cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                color = 'r', linewidth = 1)

# Creating an annotating box
annot = ax.annotate("", xy=(0,0), xytext=(-40,40),textcoords="offset points",
                    bbox=dict(boxstyle='round4', fc='linen',ec='k',lw=1),
                    arrowprops=dict(arrowstyle='-|>'))
annot.set_visible(True)

# Function for storing and showing the clicked values
#print("Import E:")
#z = 1.5  # import E
z = 0.01 # import E
#try:
# z = float(input('E: '))
# if z > 0:
 #   print('E = ',z)
#except:
 #z = 1.11
print("Default!, E = ",z)

coord = []
#initials = []
def onclick(event):
    global coord
    #global initials
    #coord.append((event.xdata, event.ydata, z))
    coord.append((event.ydata,event.xdata, z))
    x1 = event.xdata
    y1 = event.ydata
    #z = 1.61  
    # printing the values of the selected point
    print(y1,x1,z)
    annot.xy = (x1,y1)
    text = "[{:.3g}, {:.5}]".format(x1,y1)
    annot.set_text(text)
    annot.set_visible(True)
    fig.canvas.draw() #redraw the figure
try:
 task = float(input('1: Have Peak data; 2: Click Peak by mouse. Please chose number and Enter \n'))
 if task == 1:
    print('Have Peak data')
    initials=[]
    with open('initials.json', 'r') as f:
        data_peak= json.loads(f.read())
        #print(data_peak)
        initials=data_peak
        print(initials)
    plt.close()
 elif task == 2:
    print('Wait click Peak by mouse ')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    #with open("initials.txt", "w") as output:
        #output.write(str(initials))
    initials = list(coord)
    print(initials)
    with open("initials.json", "w") as output:
        output.write(json.dumps(initials))
        #output.write(str(initials))
        print(initials)
 else:
     print("No task")
except:
    print("Exit")

#vas.mpl_connect('button_press_event', onclick)
#plt.show()

#initials = list(coord)
#print(initials)

# determines the number of gaussian functions 
# to compute from the initial guesses
n_value = len(initials)

# defines a typical gaussian function, of independent variable x,
# amplitude a, position b, and width parameter c.
def gaussian(x,a,b,c):
    return a*np.exp(-np.pi*(x-b)**2 / (c**2))

# defines the expected resultant as a sum of intrinsic gaussian functions
def GaussSum(x, p, n):
    return builtins.sum(gaussian(x, p[3*k], p[3*k+1], p[3*k+2]) for k in range(n))

class NeuralNetworks(object):
    def __init__(self):  # Initializing the weight vectors
        self.n = 2  # Number of hidden neurons
        self.eta = 0.01  # Gradient step learning rate
        #self.w_1 = np.random.normal(0, 8, (self.n, 1))  # Input --> Hidden initial weight vector
        self.w_1 = np.linspace(0, 1, 2)[:, np.newaxis]
        self.b_1 = np.ones((self.n, 1))
        #self.w_2 = np.random.uniform(0, 0.01, (self.n, 1))  # Hidden --> Output initial weight vector
        self.w_2 = np.linspace(0, 1, 2)[:, np.newaxis]
        self.b_2 = np.ones((1, 1))
    
    def FeedForward(self, x, a, b, c):  # This method feeds forward the input x and returns the predicted output
        # I use the same notation as Haykin book
        self.v_1 = x * (self.w_1) + self.b_1  # Local Induced Fileds of the hidden layer
        # edit code
        self.y_1 = gaussian(self.v_1, a, b, c)  # hàm truyền

        self.v_2 = self.y_1.T.dot(self.w_2) + self.b_2 # x.T chuyển hàng thành cột, dot: vô hướng
        self.o = self.v_2  # output of the network
        
        return self.o
    
    def loss(self, x, d):  # Calculates the cost function of the network for a 'vector' of the inputs/outputs
        # x : input vector
        # d : desired output
        temp = np.zeros(len(x))
        for i in range(len(x)):
            temp[i] = d[i] - self.FeedForward(x[i])
        self.cost = np.mean(np.square(temp))
        return self.cost

    def BackPropagate(self, x, y, d):
        # Given the input, desired output, and predicted output
        # this method update the weights accordingly
        # I used the same notation as in Haykin: (4.13)
        self.delta_out = (d - y) * 1  # 1: phi' of the output at the local induced field
        self.w_2 += self.eta * self.delta_out * self.y_1
        self.b_2 += self.eta * self.delta_out

        # edit code
        self.delta_1 = (1 - np.power(np.tanh(self.v_1), 2)) * (self.w_2) * self.delta_out
        #self.delta_1 = (1 - np.power(np.maximum(self.v_1,0), 2)) * (self.w_2) * self.delta_out
        #edit code
        #self.delta_1 = (1 - np.power(np.exp(-(self.v_1)**2), 2)) * (self.w_2) * self.delta_out
        self.w_1 += self.eta * x * self.delta_1
        self.b_1 += self.eta * self.delta_1

    def train(self, x, d, epoch=100):  # Given a vector of input and desired output, this method trains the network
        iter = 0
        while (iter != epoch):
            for i in range(len(x)):
                o = self.FeedForward(x[i])  # Feeding forward
                self.BackPropagate(x[i], o, d[i])  # Backpropagating the error and updating the weights
            if iter % (epoch / 5) == 0:
                print("Epoch: %d\nLoss: %f" % (iter, self.loss(x, d)))
            iter += 1
# add code
    def findpeak(self, x, a, b, c): 
        from scipy.signal import find_peaks
        #import matplotlib.pyplot as plt
        
        # Example data
        self.max_x=np.max(x)
        self.min_x=np.min(x)
        x = np.linspace(self.min_x, self.max_x)
        #y =  gaussian(x,a,b,c)
        #y = np.squeeze(zeros(len(x)))
        y =  np.squeeze(self.FeedForward(x,a,b,c))
        #y = np.sin(x / 1000) + 0.1 * np.random.rand(*x.shape)
        
        # Find peaks
        self.i_peaks, _ = find_peaks(y)
        #print(i_peaks)
        
        # Find the index from the maximum peak
        #i_max_peak = i_peaks[np.argmax(y[i_peaks])]
        
        # Find the x value from that index
        #x_max = x[i_max_peak]
        Tm = x[self.i_peaks]
        
        # Plot the figure
        #plt.plot(x, y)
        #plt.plot(x[i_peaks], y[i_peaks], 'x')
        #plt.axvline(x=x_max, ls='--', color="k")
        #plt.show()
        return Tm
# add code
    def findpeakY(self, x, a, b, c): 
        from scipy.signal import find_peaks
        #import matplotlib.pyplot as plt
        
        # Example data
        self.max_x=np.max(x)
        self.min_x=np.min(x)
        x = np.linspace(self.min_x, self.max_x)
        #y =  gaussian(x,a,b,c)
        #y = np.squeeze(zeros(len(x)))
        y =  np.squeeze(self.FeedForward(x,a,b,c))
          
        # Find peaks
        self.i_peaks, _ = find_peaks(y)
        #print(i_peaks)
        
        # Find the index from the maximum peak
        #i_max_peak = i_peaks[np.argmax(y[i_peaks])]
        
        # Find the x value from that index
        #x_max = x[i_max_peak]
        Im = y[self.i_peaks]
        
        # Plot the figure
        #plt.plot(x, y)
        #plt.plot(x[i_peaks], y[i_peaks], 'x')
        #plt.axvline(x=x_max, ls='--', color="k")
        #plt.show()
        return Im


#--------------------
ndsang = NeuralNetworks()
#print ("Initial Loss: %f"%(ndsang.loss(x,d)))
#print("----|Training|----")
#ndsang.train(x,d,2000)
#print("----Training Completed----")

# add fit NN Gauss
def fitGauss(x,y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    gaussian_process.fit(x, y)

# defines condition of minimization, called the resudual, which is defined
# as the difference between the data and the function.
def residuals(p, y, x, n):
    return y - GaussSum(x,p,n)

# defines ffGOK=s, remove x=T, a=Im, only b=Tm, c=E
def ffGOK(x,a,b,c):
    kbz = 8.617385e-5
    hr=1
    return (hr*c)/(kbz*b**2)*np.exp(c/kbz/b)

# defines FOM
def FOM(p, y, x, n):
     return sum(y - GaussSum(x,p,n))/sum(GaussSum(x,p,n))

def E1(Tm,T1,T2):
    k = 8.617385e-5
    #return abs(1.51*((k*Tm**2)/(Tm - T1))-1.58*(2*k*Tm))
    return abs((2.52+10.2*((T2-Tm)/(T2-T1)-0.42))*((k*(Tm)**2)/(T2-T1))-(2*k*Tm))
    #return (0.976+7.3*((T2-Tm)/(T2-T1)-0.42))*((k*Tm**2)/(Tm - T1))	

# Function to calculate the TL, F1 model, T=x, maxi=a, maxt=b, engy=c
def x_function(x,a,b,c):
    #a,b,c,bv=10968.19, 473, 1.1, 1.61
    #kbz = 8.617385e-5
    return a * np.exp(-np.pi * (x - b) ** 2 / (c ** 2))
    #return a*np.exp(1.0+c/kbz/x*((x-b)/b)-((x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)-2.0*kbz*b/c)
    #return a*(bv**(bv/(bv-1.0)))*np.exp(c/kbz/x*((x-b)/b))*(((bv-1.0)*(x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)+1+(bv-1.0)*2.0*kbz*b/c)**(-bv/(bv-1.0))

# Convert decimal to a binary string
def den2bin(f):
	bStr = ''
	n = int(f)
	if n < 0: raise
	if n == 0: return '0'
	while n > 0:
		bStr = str(n % 2) + bStr
		n = n >> 1
	return bStr

#Convert decimal to a binary string of desired size of bits 
def d2b(f, b):
	n = int(f)
	base = int(b)
	ret = ""
	for y in range(base-1, -1, -1):
		ret += str((n >> y) & 1)
	return ret

#Invert Chromosome
def invchr(string, position):
	if int(string[position]) == 1:
		
		string = string[:position] + '0' + string[position+1:]
	else:
		string = string[:position] + '1' + string[position+1:]
	return string


#Roulette Wheel
def roulette(values, fitness):
	n_rand = random()*fitness
	sum_fit = 0
	for i in range(len(values)):
		sum_fit += values[i]
		if sum_fit >= n_rand:
			break
	return i	
# Func GA
def geneticTL(x,a,b,c):
    # Genetic Algorithm Code to find the Maximum of F(X)
    
    #x_function(x,a,b,c,bv)
    #Range of Values
    #x_max = 32000
    x_max = np.max(x)
    #x_min = 0
    x_min = np.min(x)
    
    #GA Parameters
    # Due my laziness to do the code, the population size must be a even number and the values for x are always integers.
    # Feel free to correct it :) 
    pop_size = 200
    mutation_probability = 0.20
    num_gen = 10
    number_of_generations = num_gen

    #Variables & Lists to be used during the code
    gen_1_xvalues = []
    gen_1_fvalues = []
    generations_x = []
    generations_f = []
    fitness = 0
    
    
    #Size of the string in bit
    x_size = int(len(den2bin(x_max)))

    #first population - random values
    for i in range(pop_size):
    	x_tmp = int(round(randint(x_max-x_min)+x_min))
    	gen_1_xvalues.append(x_tmp)
    
    	f_tmp = x_function(x_tmp,a,b,c)
    	gen_1_fvalues.append(f_tmp)
    
    	#Create total fitness
    	fitness += f_tmp
    #print ('GEN 1', gen_1_xvalues)
    
    #Getting maximum value for initial population
    max_f_gen1 = 0
    for i in range(pop_size):
    		if gen_1_fvalues[i] >= max_f_gen1:
    			max_f_gen1 = gen_1_fvalues[i]
    			max_x_gen1 = gen_1_xvalues[i]
    
    #Starting GA loop
    
    for i in range(number_of_generations):
    	#Reseting list for 2nd generation
    	gen_2_xvalues = []
    	gen_2_fvalues = []
    	selected = []
    
    	#Selecting individuals to reproduce
    	for j in range(pop_size):
    		ind_sel = roulette(gen_1_fvalues,fitness)
    		selected.append(gen_1_xvalues[ind_sel])
    
    	#Crossing the selected members
    	for j in range(0, pop_size, 2):
    		sel_ind_A = d2b(selected[j],x_size)
    		sel_ind_B = d2b(selected[j+1],x_size)
    	
    	#select point to cross over
    		cut_point = randint(1,x_size)
    	
    	#new individual AB
    		ind_AB = sel_ind_A[:cut_point] + sel_ind_B[cut_point:]
    
    	#mutation AB
    		ran_mut = random()
    		if ran_mut < mutation_probability:
    			gene_position = randint(0,x_size)
    			ind_mut = invchr(ind_AB, gene_position)
    			ind_AB = ind_mut
    	
    	#new individual BA
    		ind_BA = sel_ind_B[:cut_point] + sel_ind_A[cut_point:]		
    
    
    	#mutation BA
    		ran_mut = random()
    		if ran_mut < mutation_probability:
    			gene_position = randint(0,x_size)
    			ind_mut = invchr(ind_BA, gene_position)
    			ind_BA = ind_mut
    
    	#Creating Generation 2
    		new_AB = int(ind_AB,2)
    		gen_2_xvalues.append(new_AB)
    
    		new_f_AB = x_function(new_AB,a,b,c)
    		gen_2_fvalues.append(new_f_AB)
    
    		new_BA = int(ind_BA,2)
    		gen_2_xvalues.append(new_BA)
    
    		new_f_BA = x_function(new_BA,a,b,c)
    		gen_2_fvalues.append(new_f_BA)
    	#print ('GEN',i+2, gen_2_xvalues)
    
    
    	#Getting maximum value
    	max_f_gen2 = 0
    	for j in range(pop_size):
    		if gen_2_fvalues[j] >= max_f_gen2:
    			max_f_gen2 = gen_2_fvalues[j]
    			max_x_gen2 = gen_2_xvalues[j]
    
    	#Elitism one individual
    	if max_f_gen1 > max_f_gen2:
    		max_f_gen2 = max_f_gen1
    		max_x_gen2 = max_x_gen1
    		gen_2_fvalues[0] = max_f_gen1
    		gen_2_xvalues[0] = max_x_gen1
    	
    	#Transform gen2 into gen1
    	gen_1_xvalues = gen_2_xvalues
    	gen_1_fvalues = gen_2_fvalues
    	max_x_gen1 = max_x_gen2
    	max_f_gen1 = max_f_gen2
    	generations_x.append(max_x_gen2)
    	generations_f.append(max_f_gen2)
    
    	#Creating new fitness
    	fitness = 0
    	for j in range(pop_size):
    		f_tmp = x_function(gen_1_xvalues[j],a,b,c)
    		fitness += f_tmp
    #print ("Max xy peak:",generations_x[num_gen-1],generations_f[num_gen-1])
    Tm = generations_x[num_gen-1]
    #Im = generations_f[num_gen-1]
    return Tm
# Func GA
def geneticTLy(x,a,b,c):
    # Genetic Algorithm Code to find the Maximum of F(X)
    #Range of Values
    #x_max = 32000
    #x_max = 600
    #x_min = 0
    #x_min = 0
    #x_max = 32000
    x_max = np.max(x)
    #x_min = 0
    x_min = np.min(x)

    #GA Parameters
    # Due my laziness to do the code, the population size must be a even number and the values for x are always integers.
    # Feel free to correct it :) 
    pop_size = 200
    mutation_probability = 0.20
    num_gen = 10
    number_of_generations = num_gen

    #Variables & Lists to be used during the code
    gen_1_xvalues = []
    gen_1_fvalues = []
    generations_x = []
    generations_f = []
    fitness = 0
    
    
    #Size of the string in bit
    x_size = int(len(den2bin(x_max)))
    
    #print ("Maximum size of x is", x_max,  "characters",x_max , "variables.")
    #print ("Maximum chromosome size of x is", x_size,  "bits, i.e.,", pow(2,x_size), "variables.")
    
    
    #first population - random values
    for i in range(pop_size):
    	x_tmp = int(round(randint(x_max-x_min)+x_min))
    	gen_1_xvalues.append(x_tmp)
    
    	f_tmp = x_function(x_tmp,a,b,c)
    	gen_1_fvalues.append(f_tmp)
    
    	#Create total fitness
    	fitness += f_tmp
    #print ('GEN 1', gen_1_xvalues)
    
    #Getting maximum value for initial population
    max_f_gen1 = 0
    for i in range(pop_size):
    		if gen_1_fvalues[i] >= max_f_gen1:
    			max_f_gen1 = gen_1_fvalues[i]
    			max_x_gen1 = gen_1_xvalues[i]
    
    #Starting GA loop
    
    for i in range(number_of_generations):
    	#Reseting list for 2nd generation
    	gen_2_xvalues = []
    	gen_2_fvalues = []
    	selected = []
    
    	#Selecting individuals to reproduce
    	for j in range(pop_size):
    		ind_sel = roulette(gen_1_fvalues,fitness)
    		selected.append(gen_1_xvalues[ind_sel])
    
    	#Crossing the selected members
    	for j in range(0, pop_size, 2):
    		sel_ind_A = d2b(selected[j],x_size)
    		sel_ind_B = d2b(selected[j+1],x_size)
    	
    	#select point to cross over
    		cut_point = randint(1,x_size)
    	
    	#new individual AB
    		ind_AB = sel_ind_A[:cut_point] + sel_ind_B[cut_point:]
    
    	#mutation AB
    		ran_mut = random()
    		if ran_mut < mutation_probability:
    			gene_position = randint(0,x_size)
    			ind_mut = invchr(ind_AB, gene_position)
    			ind_AB = ind_mut
    	
    	#new individual BA
    		ind_BA = sel_ind_B[:cut_point] + sel_ind_A[cut_point:]		
    
    
    	#mutation BA
    		ran_mut = random()
    		if ran_mut < mutation_probability:
    			gene_position = randint(0,x_size)
    			ind_mut = invchr(ind_BA, gene_position)
    			ind_BA = ind_mut
    
    	#Creating Generation 2
    		new_AB = int(ind_AB,2)
    		gen_2_xvalues.append(new_AB)
    
    		new_f_AB = x_function(new_AB,a,b,c)
    		gen_2_fvalues.append(new_f_AB)
    
    		new_BA = int(ind_BA,2)
    		gen_2_xvalues.append(new_BA)
    
    		new_f_BA = x_function(new_BA,a,b,c)
    		gen_2_fvalues.append(new_f_BA)
    	#print ('GEN',i+2, gen_2_xvalues)

    	#Getting maximum value
    	max_f_gen2 = 0
    	for j in range(pop_size):
    		if gen_2_fvalues[j] >= max_f_gen2:
    			max_f_gen2 = gen_2_fvalues[j]
    			max_x_gen2 = gen_2_xvalues[j]
    
    	#Elitism one individual
    	if max_f_gen1 > max_f_gen2:
    		max_f_gen2 = max_f_gen1
    		max_x_gen2 = max_x_gen1
    		gen_2_fvalues[0] = max_f_gen1
    		gen_2_xvalues[0] = max_x_gen1
    	
    	#Transform gen2 into gen1
    	gen_1_xvalues = gen_2_xvalues
    	gen_1_fvalues = gen_2_fvalues
    	max_x_gen1 = max_x_gen2
    	max_f_gen1 = max_f_gen2
    	generations_x.append(max_x_gen2)
    	generations_f.append(max_f_gen2)
    
    	#Creating new fitness
    	fitness = 0
    	for j in range(pop_size):
    		f_tmp = x_function(gen_1_xvalues[j],a,b,c)
    		fitness += f_tmp
    #print ("Max xy peak:",generations_x[num_gen-1],generations_f[num_gen-1])
    #Tm = generations_x[num_gen-1]
    Im = generations_f[num_gen-1]
    return Im
# using least-squares optimization, minimize the difference between the data
import math
def AE_gen(x,a,b,c):
    #x = data_set[:,0]
    y=gaussian(x,a,b,c)
    #Tm = geneticTL(x,a,b,c)
    #Im = geneticTLy(x,a,b,c)
    Tm = ndsang.findpeak(x,a,b,c)
    Im = ndsang.findpeakY(x,a,b,c)
   
    #
    y_half=np.zeros_like(x)+Im/2
    #print("Img=",Im)

    #
    idx = np.argwhere(np.diff(np.sign(y-y_half))).flatten()

    #print("T1,T2",  x[idx])
    # Calculate E, method PS
    T1=x[idx][0]
    T2=x[idx][1]
    #
    # add beta = FWHM (rad), lamda (A0)
    kz = 0.89
    #kz = 0.98
    lamda = 0.15405
    FWHM=(T2-T1)*math.pi/360
    print("FWHM=",FWHM)
    costheta=np.cos(Tm*math.pi/360)
    tantheta=np.tan(Tm*math.pi/360)
    Di=kz*abs(lamda/(FWHM*costheta))
    print("Di=",Di)
    Esi=abs(FWHM/(4*tantheta))
    print("Esi=", Esi)
    return Di


# define the true objective function IR
def IR(x, E, b):
    return E * x + b
# E: IR
def AE_IR(x,a,b,c):
    #x = data_set[:,0]
    y=gaussian(x,a,b,c)
    #Tm = geneticTL(x,a,b,c)
    #Im = geneticTLy(x,a,b,c)
    Tm = ndsang.findpeak(x,a,b,c)
    Im = ndsang.findpeakY(x,a,b,c)

    #Tm = generations_x[num_gen-1]
    #Im = generations_f[num_gen-1]
    
 #add IR
    #j=15
    j=17
    Tci=[]
    Ici=[]
    #Ici=[1]
    
    for i in range(1,j):
        #Im[i]=geneticTLy(x,a,b,c)
        yi=np.zeros_like(x)+Im*i/100
        idx = np.argwhere(np.diff(np.sign(y - yi))).flatten()
        Tc = x[idx][0]
        Ic = y[idx][0]
        #print("Tc,Ic=",Tc,Ic)
        Tci=np.append(Tci, Tc)
        Ici=np.append(Ici, Ic)

        # add beta = FWHM2 (rad), lamda (A0)
    
    kz = 0.89
    #kz = 0.9
    #lamda = 1.5406
    lamda = 0.15405
    costheta2 = np.cos(Tm*math.pi/360)
    tantheta2 = np.tan(Tm*math.pi/360)
    #kbz = 8.617385e-5
    # ND
    #x_ND=1/(kbz*Tci)

    # FWHM
    #x_ND = Tci*sqrt(2)
    #x_ND = Tci/(2*costheta2/kz/lamda)
    x_ND = Tci
    
    # ln(TL)
    y_ln=np.log(Ici)
    
    # curve fit
    popt, _ = curve_fit(IR, x_ND, y_ln)
    # summarize the parameter values
    E_IR, b_IR = popt
    #print("HSG a & b:",-E_IR,b_IR)
    #print('y = %.5f * x + %.5f', % (E_IR,b_IR))
    #FWHM2 = E_IR
    
    #FWHM2 = 1/(E_IR*np.sqrt(log(4)))
    #β = 0.5 H (π / loge2)1/2 -> H = 2.β /(π / loge2)1/2
    # H=FWHM2, β = 1/E_IR
    #FWHM2 = 2*(1/E_IR)/(np.sqrt(np.pi/log(2)))
    # E_IR = 1/FWHM  peak width
    


    #Di2 = kz * abs(lamda / (FWHM2 * costheta2))
    Di2 = E_IR
    print("Di_IR=", Di2)
    
    FWHM2 = kz * abs(lamda / (costheta2))/E_IR
    #print("FWHM2 = ",FWHM2)
    
    Esi2 = abs(FWHM2 / (4 * tantheta2))
    print("Esi_IR=", Esi2)
    
    #define function to calculate adjusted r-squared
    #def R2(x1_ND, y1_ln, degree):
    #results = {}
    coeffs = np.polyfit(x_ND, y_ln, 1)
    p = np.poly1d(coeffs)
    yhat = p(x_ND)
    ybar = np.sum(y_ln)/len(y_ln)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y_ln - ybar)**2)
    R2 = 1- (((1-(ssreg/sstot))*(len(y_ln)-1))/(len(y_ln)-1))
    #print("R2=",R2)

    # Create linear regression object
    #regr = linear_model.LinearRegression()
    # Make predictions using the testing set
    y_pred = x_ND*E_IR + b_IR

    # format and show
    fig = plt.figure()
    bx = fig.subplots()
    # sets the axis labels and parameters.
    bx.tick_params(direction='in', pad=15)
    bx.set_xlabel('$2θ$ ($^o$)', fontsize=15)
    bx.set_ylabel('lnI', fontsize=15)
    #bx.plot(x_ND, y_ln, 'r-')

    bx.scatter(x_ND, y_ln, color='blue', label='Data')

    bx.text(0.1, 0.70, r'Di_IR = {0:0.6f}'
             .format(E_IR), transform=bx.transAxes)
    #add code
    bx.text(0.1, 0.02, r'y = {0:0.5f}*x+ {1:0.5f}, R$^2$ = {2:0.5f}'
            .format(E_IR, b_IR, R2), transform=bx.transAxes)

    #bx.text(0.02, 0.65, r'R$^2$ = {0:0.6f}'
           # .format(R2), transform=bx.transAxes)
    # add code
    bx.plot(x_ND, y_pred, color="red", linewidth=3)

    #plt.tight_layout()
    #fig.savefig(r'GA_XRD.png')
    #plt.show()

    return Di2

cnsts =  leastsq(
            #geneticTL,
            residuals, 
            initials, 
            args=(
                data_set[:,1],          # y data
                data_set[:,0],          # x data
                n_value                 # n value
            )
        )[0]

# integrates the gaussian functions through gauss quadrature and saves the 
# results to a list, and each list is saved to its corresponding data file 

# Create figure window to plot data
fig = plt.figure(1, figsize=(9.5, 6.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[6, 2])

#plt.title('ANN approach for analyzing TL curve by Gaussian model')
# Top plot: data and fit
ax1 = fig.add_subplot(gs[0])

#sets the axis labels and parameters.
ax1.tick_params(direction = 'in', pad = 15)
#ax1.set_xlabel('Temperature (K)', fontsize = 15)
ax1.set_ylabel('Intensity  (a.u.)', fontsize = 15)

# plots the first two data sets: the raw data and the GaussSum.
ax1.plot(data_set[:,0], data_set[:,1], 'ko')
ax1.plot(x,GaussSum(x,cnsts, n_value))

# adds a plot
#ax1.fill_between(x, GaussSum(x,cnsts, n_value), facecolor="yellow", alpha=0.25)of each individual gaussian to the graph.
for i in range(n_value):
    ax1.plot(
        x, 
        gaussian(
            x, 
            cnsts[3*i],
            cnsts[3*i+1],
            cnsts[3*i+2]
            #cnsts[4*i+3]
        )
    )
# adds color a plot of each individual gaussian to the graph.
for i in range(n_value):
    ax1.fill_between(
        x, 
        gaussian(
            x, 
            cnsts[3*i],
            cnsts[3*i+1],
            cnsts[3*i+2]
            #cnsts[4*i+3]
        ),alpha=0.25
    )
# adds a ffGOK of each individual gaussian to the graph.
AE2 = dict()
for i in range(n_value):
    AE2[i] = AE_gen(
            x,
            cnsts[3*i],
            cnsts[3*i+1],
            cnsts[3*i+2]
            #cnsts[4*i+3]
        )

"""    
# adds a ffGOK of each individual gaussian to the graph.
AE3 = dict()
for i in range(n_value):
    AE3[i] = AE_IR(
            x, 
            cnsts[3*i],
            cnsts[3*i+1],
            cnsts[3*i+2]
            #cnsts[4*i+3]
        )
"""
# adds a ffGOK of each individual gaussian to the graph.
GA1 = dict()
for i in range(n_value):
    GA1[i] = geneticTL(
            x, 
            cnsts[3*i],
            cnsts[3*i+1],
            cnsts[3*i+2]
           # cnsts[4*i+3]
    )
GA2 = dict()
for i in range(n_value):
    GA2[i] = geneticTLy(
            x, 
            cnsts[3*i],
            cnsts[3*i+1],
            cnsts[3*i+2]
            #cnsts[4*i+3]
        )

# creates ledger for each graph
ledger = ['Data', 'Resultant']

DA=[]

for i in range(n_value):
    ledger.append(
        'P' + str(i+1)
                + ', D' + str(i+1) +' = '+ str(AE2[i])+ ' nm'
        #+ ', D' + str(i+1) +' = ' +str(round(AE2[i][0],3)) + ' nm'
        #+ ', E' +'$_{PS}$'+ str(i+1) +' = '+ str(round(AE2[i][0],3)) + ' eV'
         #+ ', t'+ str(i+1) +'$_{1/2}$ = ' + str(round(half1[i],3))+ ' s'
        #+ ', Tm'  + str(i+1) +' = '+ str(round(GA1[i],3)) + ' K'
        #+ ', Im'  + str(i+1) +' = '+ str(round(GA2[i],3)) + ' a.u.'
        #+ ', s = ' + str(round(ff[i])) + ' s$^{-1}$'        xxmxmxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        #+ '\ns = ' + str(round(ff[i]))[0] + '.' + str(round(ff[i]))[1] + str(round(ff[i]))[2] + 'x$10^{15}$' + ' s$^{-1}$'
    ) 
    DA = np.append(DA,AE2[i])
    
print("DA=",np.sum(DA)/n_value," nm")
#adds the ledger to the graph.
ax1.legend(ledger)

#adds text FOM
#ax1.text(0.01, 0.25, r'GOK''\n FOM = {0:0.4f}'
ax1.text(0.01, 0.995, r'Gaussuan model ANN'+', FOM = {0:0.6f}'
         .format(abs(FOM(cnsts, y, x, n_value))) + ', DA_GS_PS_ANN = {0:0.6f} nm\n'
         .format(np.sum(DA) / n_value), transform=ax1.transAxes)
#ax1.text(0.01, 0.25, r'FOM = {0:0.9f}'
         #.format(abs(FOM(cnsts, y, x, n_value))), transform=ax1.transAxes)
#ax1.text(0.80, 0.99, r'DA = {0:0.6f} nm'
         #.format(np.sum(DA)/n_value), transform=ax1.transAxes)

# Bottom plot: residuals
ax2 = fig.add_subplot(gs[1])
ax2.plot(x,residuals(cnsts, y, x, n_value))
#ax2.plot(x,GaussSum(x,cnsts, n_value))
#plt.title('ANN approach for analyzing TL curve by Gaussian model')

ax2.set_xlabel('Temperature (K)',fontsize = 15)
ax2.set_ylabel('Residuals', fontsize = 15)
#ax2.set_ylim(-20, 20)
#ax2.set_yticks((-20, 0, 20))

# format and show
plt.tight_layout()
# format and show
#plt1.tight_layout()
#plot(fitGauss)

fig.savefig(r'ANN_FOK.png')

plt.show()

# add code
data_new = {'T':x,'I':GaussSum(x,cnsts, n_value)}
# save to csv
df = pd.DataFrame(data_new)
# saving the dataframe
#df.to_csv('C1.csv', sep=';', header=False, index=False)
df.to_csv('TL_FOK_new.csv',header=False, index=False)

#X= np.linspace(0, 600, 12000)[:, np.newaxis]
#y = np.squeeze(GaussSum(x,cnsts, n_value))

'''
#X= np.linspace(0, 700, 200)[:, np.newaxis]
X= np.linspace(min_x, max_x, 200)[:, np.newaxis]
#X= np.array([min_x, max_x]).reshape(1, -1)
y1 = np.squeeze(GaussSum(X,cnsts, n_value))

rng = np.random.RandomState(0)
training_indices = rng.choice(np.arange(y1.size), size=1600)
X_train, y_train = X[training_indices], y1[training_indices]

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

#plt.plot(X, y1, label=r"$f(x) = Gauss(x)$", linestyle="dotted", c='g')
plt.scatter(X_train, y_train, label="Observations", c='b')
plt.scatter(x, y, label="Data",s= 20, c='black')
plt.plot(X, mean_prediction, label="Mean prediction", c='r')
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.4,
    label=r"95% confidence interval",
)
#plt.show()
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
#_ = plt.title("Gaussian process regression")
plt.show()
'''
# close window tinker
#destroy()
sys.exit(0)
#del x, X
#del y
