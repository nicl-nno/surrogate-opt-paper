import time

from hyperopt import fmin, tpe, space_eval
from hyperopt import hp

from src.basic_evolution.main import run_genetic_opt

space = hp.choice('a',
                  [
                      (hp.randint("max_gens", 15) * 2 + 6, hp.randint("pop_size", 20) + 10,
                       hp.randint("sur_points", 500) + 10,
                       hp.randint("initial_fidelity_time", 500) + 10,
                       hp.randint("initial_fidelity_spatial", 500) + 10)

                  ])

score_history = []
best_score_history = []

best_score = 99999


def objective(args, criteria_id):
    max_gens, pop_size, archive_size_rate, crossover_rate, mutation_rate, p1, p2, p3 = args

    print(time.strftime("%Y-%m-%d %H:%M"))

    print("OBJ", max_gens, pop_size, archive_size_rate, crossover_rate, mutation_rate, p1, p2, p3)
    archive_size = round(archive_size_rate * pop_size)
    score = run_robustess_exp(max_gens, pop_size, archive_size, crossover_rate, mutation_rate, [p1, p2, p3])

    if (len(best_score_history) == 0 or score[criteria_id] < best_score_history[len(best_score_history) - 1][0][
        criteria_id]):
        best_score = score
        best_score_history.append([best_score, ])
    score_history.append([score, ])

    return score[criteria_id]


def objective_robustparams(args):
    return objective(args, 3)


def objective_q(args):
    return objective(args, 2)


def objective_tradeoff(args):
    return objective(args, 1)


def objective_metrics(args):
    return objective(args, 0)


print("START OPT")
best = fmin(objective_robustparams, space, algo=tpe.suggest, max_evals=100, verbose=True)
print("END OPT")
print(space_eval(space, best))

for item in best_score_history:
    print(item[0:(len(item) - 1)])

print(best)
