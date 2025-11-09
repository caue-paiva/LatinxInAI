import math, random
import matplotlib.pyplot as plt



def sine_data(start:int, end:int, delta:int) ->list[float]:
    """
    Generates  sine function data from starting to ending integer in discrete delta steps  
    """
    result = []

    for i in range(start,end+1,delta):
        result.append(math.sin(i))
    
    return result


def measurement_data(start:int, end:int, delta:int) ->list[int]:
    """
    Generates data from starting to ending integer in discrete delta steps  
    """
    return [i for i in range(start,end+1,delta)]

def A(x): #in the 1D form, the state transition matrix is just a function that adds 0.1 to the state (x axis) 
    return x + 0.1

def H(x): #measurement 'matrix' = sine function
    return math.sin(x)

q = 0.0 #state transition noise
r = 0.1 # measurement noise


#data = measurement_data(0,100,1)

x0 = 0
x:float = 0
y:float

# Calculate perfect measurement (no noise)
y_perfect = H(x)
    
# Calculate upper and lower bounds
y_upper = H(x) + r
y_lower = H(x) - r

results:list = [(x, y_perfect, y_upper, y_lower)] # list of tuples (x, y_perfect, y_upper, y_lower)

for _ in range(100):
    
    x = A(x) + q #get next state
    
    # Calculate perfect measurement (no noise)
    y_perfect = H(x)
    
    # Calculate upper and lower bounds
    y_upper = H(x) + r
    y_lower = H(x) - r

    results.append((x, y_perfect, y_upper, y_lower))

# Plot the results
x_values = [result[0] for result in results]
y_perfect = [result[1] for result in results]
y_upper = [result[2] for result in results]
y_lower = [result[3] for result in results]

plt.figure(figsize=(12, 7))

# Plot the probability region (shaded area between y_upper and y_lower)
plt.fill_between(x_values, y_lower, y_upper, alpha=0.3, color='lightblue', 
                 label='Probability Region (±r)')

# Plot the perfect measurement line
plt.plot(x_values, y_perfect, 'b-', linewidth=2, label='Perfect Measurement (no noise)')

# Plot the upper and lower bounds
plt.plot(x_values, y_upper, 'r--', linewidth=1, alpha=0.7, label='Upper Bound (+r)')
plt.plot(x_values, y_lower, 'g--', linewidth=1, alpha=0.7, label='Lower Bound (-r)')

plt.xlabel('State (x)', fontsize=12)
plt.ylabel('Measurement (y)', fontsize=12)
plt.title('Kalman Filter 1D: Measurement with Probability Region', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


