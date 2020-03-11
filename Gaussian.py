#Import the plotter library
import matplotlib.pyplot as plt
#Import numpy library for arrays
import numpy as np

"""
The following code does the following:
    1) Creates a test function to give you sample data.
        x - corresponds to the money put into an advertisement
        y - corresponds to the number of products sold
    2) creates an estimator for the test function using a 4th order
        polynomial least squares
    3) Creates an estimator for the standard deviation of the function.
    4) Finds the point of maximum product sales/$
        - Finds the standard deviation at this point
    5) Finds the maximum point for best/worst case scenarious that will account for
        95% of the data.
        
    Disclaimer: This does not use Baysean inferance.  In order to properly make
    predictions about the future, you will need a baysean model.  Doing such a model 
    was beyond the scope given the amount of time I had.  This current model should
    serve as a guideline and hopefully give a somewhat decent result (and point you in the)
    right direction.  Please double check my math before you use this for any big business decisions.
    
    Love you.
"""

#Create a line from 0-10 incrementing by .1
x = np.arange(0,10,.1)
#Create a test function (the cubed root of x)
f_signal = x**.33

#A function to add noise to our data.
def add_noise(y):
    #Creates gaussian random number (centered at 0 with a std of .075)
        #Scale this random number by the square of y to create noise. Then add noise
        #to y.
    epsilon = np.random.normal(0,.075,(y.shape))*y**2
    return y+epsilon

#Create a noisy signal
f = add_noise(f_signal)
#plot the noisy signal
plt.scatter(x,f,label='noisy function')
plt.legend()
plt.show()


def kde_plot(y,window_size=10):
    """Creates a plot that represents the local standard deviation at a given 
    value of x.  This works by centering a window at each value of x.  The standard
    deviation of all values within this window is calculated.  Then slide the window.
    This will give the std for each value of x.  
    
    Inputs: y - The y value of your function (from which std will be calculated)
        window_size - The number of values used in the window.  
    """
    #Padd the ends with zeros so that the window will not go past the data
    padding = np.zeros(window_size)
    y_padded = np.concatenate((padding,y,padding))
    
    
    results = []
    #Iterate through each non_padded value of y.
    for i in range(window_size,len(y)+window_size):
        window = []
        #Create a list "window" consisting of the values imediately surrounding
            #y[i].
        for j in range(-window_size,window_size+1):
            window.append(y_padded[i+j])
        #Conver the window to an array (for python purposes)
        window = np.array(window)
        #Calculate the std of the window.
        std = np.std(window)
        #Append the results to a list (corresponds to element i).
        
        results.append(std)
    #Convert the results to an array and return.
    return np.array(results)


def polynomial(x):
    #A polynomial kernel for calculating least squares
    return np.stack((x**4,x**3,x**2,x,np.ones(x.shape)),1)
    
def derivative(x):
    #A way to take the derivative of an array. This is optimized for python, It
        #is hard to explain how this works, but it is basically the derivative.
    derivative_kernel = np.array([-1,0,1])
    return np.convolve(x,derivative_kernel,'same')

#################################################

"""
Use a least squares fit to find a 4th order polynomial which estimates the original 
noisy function f.
"""
#Create your polynomials (matrix is (n,5))
X = polynomial(x)

#Calculate the coefficients by least squares.  This is done by the formula:
    #A = (((X^T)(X))^-1)(X^T)(f)
coefficients = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),f)

#Just to make sure that the polynomial estimation is correct, let's plot it.
f_extimated = np.dot(X,coefficients)
plt.scatter(x,f,label='points')

plt.scatter(x,np.dot(X,coefficients),label='best fit')
plt.legend()
plt.show()
###
"""
Now let's assume that our estimator will go beyond the data.  (Remember that an
estimator will get more inaccurate the further away from the data poitns you get.
You can calculate the amount of deviation the same way that you would calculate the
deviation in a Taylor series)  For now, let's pretend that the estimator does not
deviate too much for the range that we are looking at.
"""
#Create an array that goes from 0-20 (instad of 0-10 as was done previously.)
x_extended = np.arange(0,20)
#Create a polynomial of this extended array
XX = polynomial(x_extended)
#Create an estimator of this extended array.
f_extended = np.dot(XX,coefficients)
########################################      

"""
Calculate the standard deviation about each point. This will allow us to see
how the variance changes over the course of the graph. Note: Only consider this
if the standard deviation seems to follow some sort of pattern. The function used
was deliberately selected so that the std changes with x.  
"""
#Calculate the standard deviation about each point.
std = kde_plot(f)
#Plot noisy function an std dev plot.
plt.scatter(x,f,label='points')
plt.plot(x,std,c='r',label='stdev')
plt.legend()
plt.show()
#######################################################
"""
Since we want to find the local maximum, we want to look for the point where
the derivative of f_extended is zero (or close to it)
"""
#Take derivative of f_extended
df_extended = derivative(f_extended)

#Our desired point is going to be the point closest to zero. We will ind the index of this
desired_index = np.argmin(abs(df_extended))

#Obtain the x and y coordinates of this index.
best_x = x_extended[desired_index]
best_y = f_extended[desired_index]
##################################################
"""We want to see what the standard deviation is when we are at this point. 
We will estimate this standard deviation function again using least squares and extending
the estimator. 
"""
#Calculate least squares
std_coeff = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),std)
#Calculate estimator
std_estimator = np.dot(X,std_coeff)
#Plot to check our sanity
plt.scatter(x,std,c='r',label='stdev of points')
plt.scatter(x,std_estimator,label='estimated std')
plt.legend()
plt.show()

#Extend the std function to a range of 0-20
std_extended = np.dot(XX,std_coeff)

#Obtain the std at this point.
std_at_best_x = std_extended[desired_index]

print("Best point")
print("x:      ",best_x)
print("y:      ",best_y)
print("std at x: ",std_at_best_x)


##########################################################
"""
We want to get this point again assuming both the worst case (bottom) and best case
(top) scenarios.  This is done by adding +/- 2 std to the line of best bit. 
In theroy, 95% of the data will fit between these two scenarious
"""
top_df_extended = derivative(f_extended+2*std_extended)
bottom_df_extended = derivative(f_extended-2*std_extended)

top_desired_index = np.argmin(abs(top_df_extended))
bottom_desired_index = np.argmin(abs(top_df_extended))


#Obtain the x and y coordinates of this index.
bottom_best_x = x_extended[bottom_desired_index]
bottom_best_y = f_extended[bottom_desired_index]-2*std_extended[bottom_desired_index]

top_best_x = x_extended[top_desired_index]
top_best_y = f_extended[top_desired_index]+2*std_extended[top_desired_index]

print("Best point at -2 std")
print("x:     ",bottom_best_x)
print("y:     ",bottom_best_y)

print("Best point at +2 std")
print("x:     ",top_best_x)
print("y:     ",top_best_y)








