import matplotlib.pyplot as plt
from math import exp

T_0 = 22    # Temperature in C
T_INIT = 83

TIME_STEP = 60 # time in sec.
TIME_END = 15*60

GAMMA = 0.00064

DATA_REF = [83.0, 77.7, 75.1, 73.0, 71.1, 69.4, 67.8, 66.4, 64.7, 63.4, 62.1, 61.0, 59.9, 58.7, 57.8, 56.6]

def solve(T_0, T_init, gamma, dt, t_f):
    n = int(t_f/dt)
    T_ls = list()
    T_ls.append(T_init)
    for i in range(n):
        g = -gamma*(T_ls[-1] - T_0)
        T_ls.append(T_ls[-1] + g*dt)
    return [[i*dt for i in range(len(T_ls))], T_ls]

def analit_solve(T_0, T_init, gamma, dt, t_finit):
    n = int(t_finit/dt)
    T = lambda t: (T_init - T_0) * exp(-gamma * t) + T_0
    t_ls = [i*dt for i in range(n)]
    T_ls = map(T, t_ls)
    return [t_ls, T_ls]

if __name__ == '__main__':
    dataX1, dataY1 = solve(T_0, T_0+61, GAMMA, TIME_STEP, 60*60)
    plt.plot(dataX1, dataY1, 'r')
    plt.show()
    print 5
