# import library
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
#import matplotlib.pyplot as plt1
import matplotlib.gridspec as gridspec  # unequal plots
from scipy.optimize import leastsq
from pylab import *

from matplotlib.widgets import Cursor

import os
import tkinter as tk
from tkinter import filedialog
import sys
import numpy
import math

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
ax.set_xlabel('$2θ$$^o$', fontsize = 15)
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
#
#z = 0.1e-1 # import D
z = 0.1
#try:
# z = float(input('E: '))
# if z > 0:
#    print('E = ',z)
#except:
 #z = 1.11
# print("Default!, E = ",z)

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

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

initials = list(coord)
print(initials)

# determines the number of gaussian functions 
# to compute from the initial guesses
n_value = len(initials)

import math
# defines a typical lorentz function, of independent variable x,
def lorentz(x,a,b,c):
    #return a*np.exp(-np.pi*(x-b)**2 / (c**2))
    return a * c ** 2 / ((x - b) ** 2 + (c ** 2))

# defines the expected resultant as a sum of intrinsic gaussian functions
def GaussSum(x, p, n):
    return builtins.sum(lorentz(x, p[3*k], p[3*k+1], p[3*k+2]) for k in range(n))

# defines condition of minimization, called the resudual, which is defined
# as the difference between the data and the function.
def residuals(p, y, x, n):
    return y - GaussSum(x,p,n)


# defines FOM
def FOM(p, y, x, n):
     return sum(y - GaussSum(x,p,n))/sum(GaussSum(x,p,n))
	

# Function to calculate the TL, F1 model, T=x, maxi=a, maxt=b, engy=c
def x_function(x,a,b,c):
    return a * c ** 2 / ((x - b) ** 2 + (c ** 2))
    #return a*np.exp(-np.pi*(x-b)**2 / (c**2))

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
    x_max = np.max(x)
    x_min = np.min(x)
    
    #GA Parameters
    # Due my laziness to do the code, the population size must be a even number and the values for x are always integers.
    # Feel free to correct it :) 
    pop_size = 200
    mutation_probability = 0.20
    num_gen = 20
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
    x_max = np.max(x)
    # x_min = 0
    x_min = np.min(x)
    
    #GA Parameters
    # Due my laziness to do the code, the population size must be a even number and the values for x are always integers.
    # Feel free to correct it :) 
    pop_size = 200
    mutation_probability = 0.20
    num_gen = 20
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
kz = 0.89
lamda = 0.15405
def AE_gen(x,a,b,c):
    #x = data_set[:,0]
    y=lorentz(x,a,b,c)
    Tm = geneticTL(x,a,b,c)
    Im = geneticTLy(x,a,b,c)
   
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
    #kz = 0.94
    #kz = 0.89
    #lamda = 0.15405
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
Tci = []
Ici = []
def AE_IR(x,a,b,c):
    #x = data_set[:,0]
    y=lorentz(x,a,b,c)
    Tm = geneticTL(x,a,b,c)
    Im = geneticTLy(x,a,b,c)
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
    #kz = 0.94
    kz = 0.89
    lamda = 0.15405
    FWHM2=(T2-T1)*math.pi/360
    print("FWHM2=",FWHM2)
    costheta2=np.cos(Tm*math.pi/360)
    sintheta2 = np.sin(Tm * math.pi / 360)
    #costheta2=np.cos(Tm/2)
    print("costheta2=",costheta2)
    tantheta2=np.tan(Tm*math.pi/360)
    #Di2=kz*abs(lamda/(FWHM2*costheta2))
    #print("Di2=",Di2)
    #Esi=abs(FWHM2/(4*tantheta2))
    #print("Esi=", Esi)
    #add IR
    #j=15
    #j=12
    global Tci
    global Ici
    #for i in range(1,n_value):
    #Tc = []
    #Ic = []
    #Tc.append(np.log(FWHM2))
    Tc=FWHM2*costheta2
    #Tc=FWHM2
    #print(Tc)
    #Tci=Tc
    #print(Tci)
    #Ic.append(np.log(1/costheta2))
    Ic=4*sintheta2
    #Ic=1/costheta2
    print(Ic)
    #Ici=Ic
    #print(Ici)
    Tci=np.append(Tci, Tc)
    Ici=np.append(Ici, Ic)
    #return Di2
print("Tci = ",Tci)
print("Ici = ",Ici)

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

# Top plot: data and fit
ax1 = fig.add_subplot(gs[0])

#sets the axis labels and parameters.
ax1.tick_params(direction = 'in', pad = 15)
ax1.set_xlabel('$2θ$$^o$', fontsize = 15)
ax1.set_ylabel('Intensity (a.u.)', fontsize = 15)

# plots the first two data sets: the raw data and the GaussSum.
ax1.plot(data_set[:,0], data_set[:,1], 'ko')
ax1.plot(x,GaussSum(x,cnsts, n_value))

# adds a plot
#ax1.fill_between(x, GaussSum(x,cnsts, n_value), facecolor="yellow", alpha=0.25)of each individual gaussian to the graph.
for i in range(n_value):
    ax1.plot(
        x, 
        lorentz(
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
        lorentz(
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
        )

# adds a IR of each individual gaussian to the graph.
AE3 = dict()
for i in range(n_value):
   AE3[i] = AE_IR(
            x,
            cnsts[3*i],
            cnsts[3*i+1],
            cnsts[3*i+2]
        )
#def AE_IR_DA(x,a,b,c):
#def AE_IR_DA(x_ND,y_ln ):
#print("Tci,Ici=",Tci,Ici)
x_ND = Ici
y_ln = Tci

# curve fit
popt, _ = curve_fit(IR, x_ND, y_ln)
# summarize the parameter values
E_IR, b_IR = popt
# add beta = FWHM2 (rad), lamda (A0)

#DA2 = kz * lamda / E_IR
DA2 = kz * lamda / b_IR
#DA2 = kz * lamda / b_IR
#Di2 = kz * abs(lamda / (FWHM2 * costheta2))
print("DA_LR_WH=", DA2, " nm")
#Esi2 = abs(FWHM2 / (4 * tantheta2))
#print("Esi_IR=", Esi2)

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
fig2 = plt2.figure()
bx = fig2.subplots()
# sets the axis labels and parameters.
#bx.tick_params(direction='in', zpad=15)
bx.set_xlabel('4sin$θ$', fontsize=15)
bx.set_ylabel('$β$cos$θ$', fontsize=15)
    #bx.plot(x_ND, y_ln, 'r-')

bx.scatter(x_ND, y_ln, color='blue', label='Data')

bx.text(0.1, 0.70, r'Slope = {0:0.6f}'
    .format(E_IR), transform=bx.transAxes)
    #add code
bx.text(0.1, 0.02, r'y = {0:0.5f}*x+ {1:0.5f}, R$^2$ = {2:0.5f}'
     .format(E_IR, b_IR, R2), transform=bx.transAxes)
bx.text(0.1, 0.65, r'DA_LR_WH = {0:0.6f} nm'
     .format(DA2), transform=bx.transAxes)

    #bx.text(0.02, 0.65, r'R$^2$ = {0:0.6f}'
           # .format(R2), transform=bx.transAxes)
    # add code
bx.plot(x_ND, y_pred, color="red", linewidth=3)

    #return DA2
# creates ledger for each graph
ledger = ['Data', 'Resultant']
DA1=[]
#DA2=[]
for i in range(n_value):
    ledger.append(
        'P' + str(i+1)
        + ', D'+'$_{PS}$' + str(i+1) +' = '+ str(round(AE2[i],2)) + ' nm'
        #+ ', D'+'$_{IR}$' + str(i+1) +' = ' +str(round(AE3[i],2)) + ' nm'
        #+ ', D'+'$_{IR}$' + str(i+1) +' = ' +str(round(AE3[i],2)) + ' nm'
           ) 
    #print("D"+ str(i+1)+"=",AE2[i])
    DA1 = np.append(DA1,AE2[i])
    #DA2 = np.append(DA2, AE3[i])
#AE_IR_DA(Tci,Ici)
print("DA_LR_PS=",np.sum(DA1)/n_value," nm")
#print("DA_IR=",np.sum(DA2)/n_value," nm")
#print("DA_IR=",DA2," nm")

#adds the ledger to the graph.
ax1.legend(ledger)

#adds text FOM

ax1.text(0.01, 0.995, r'Lorentz model'+', FOM = {0:0.6f}'
         .format(abs(FOM(cnsts, y, x, n_value)))+', DA_LR_PS = {0:0.6f} nm\n'
         .format(np.sum(DA1)/n_value), transform=ax1.transAxes)
#ax1.text(0.01, 0.995, r'DA_PS = {0:0.6f} nm'
         #.format(np.sum(DA1)/n_value), transform=ax1.transAxes)
#ax1.text(0.31, 0.995, r'DA_IR = {0:0.6f} nm'
         #.format(np.sum(DA2)/n_value), transform=ax1.transAxes)

# Bottom plot: residuals
ax2 = fig.add_subplot(gs[1])
ax2.plot(x,residuals(cnsts, y, x, n_value))
#ax2.plot(x,GaussSum(x,cnsts, n_value))
ax2.set_xlabel('$2θ$$^o$',fontsize = 15)
ax2.set_ylabel('Residuals', fontsize = 15)

# format and show
plt.tight_layout()
# format and show
#plt1.tight_layout()

fig.savefig(r'GA_XRD.png')

plt.show()
#plt1.show()

# add code
data_new = {'T':x,'I':GaussSum(x,cnsts, n_value)}
# save to csv
df = pd.DataFrame(data_new)
# saving the dataframe
#df.to_csv('C1.csv', sep=';', header=False, index=False)
df.to_csv('XRD_new.csv',header=False, index=False)

# close window tinker
#destroy()
sys.exit(0)

