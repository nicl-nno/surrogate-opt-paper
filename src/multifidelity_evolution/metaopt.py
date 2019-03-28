import numpy as np
from scipy.optimize import minimize, differential_evolution
from src.multifidelity_evolution.kriging_evolution import optimize_test

score_history = []
best_score_history = []

time_cons=100

def objective(args):
    max_gens, pop_size, sur_points = args
    max_gens=int(round(max_gens))
    pop_size=int(round(pop_size))
    sur_points=int(round(sur_points))


    print("OBJ", max_gens, pop_size, sur_points)
    archive_size = round(pop_size/2)

    score = optimize_test(train_stations=[1], max_gens=max_gens, pop_size=pop_size, archive_size=archive_size,
                  crossover_rate=0.7, mutation_rate=0.7, mutation_value_rate=[0.1, 0.01, 0.001], plot_figures=False, sur_points=sur_points)

    if (len(best_score_history) == 0 or score < best_score_history[len(best_score_history) - 1][
        0]):
        best_score = score
        best_score_history.append([best_score, ])
    score_history.append([score, ])


    print(score)
    return score

def constraint1(args):
    max_gens, pop_size, sur_points = args
    return (sur_points+max_gens*pop_size)-time_cons

n = 3
x0 = np.zeros(n)
x0[0] = 10
x0[1] = 10
x0[2] = 10

# show initial objective
#print('Initial SSE Objective: ' + str(objective(x0)))

# optimize
b1 = (10,50)
b2 = (5,20)
b3 = (5,20)
bnds = (b1, b2, b3)
con1 = {'type': 'eq', 'fun': constraint1}
cons = ([con1])
solution = minimize(objective,x0,method='SLSQP', bounds=bnds,constraints=cons)#,options={'maxiter': 1000})
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('x1 = ' + str(x[0]))
print('x2 = ' + str(x[1]))
print('x3 = ' + str(x[2]))