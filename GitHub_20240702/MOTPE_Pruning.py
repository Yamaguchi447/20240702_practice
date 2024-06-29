# 多目的でプルーニング

from yahpo_gym import benchmark_set
from yahpo_gym.benchmarks import *
import ConfigSpace as CS
import optuna
from optuna.samplers import MOTPESampler
from functools import partial
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from yahpo_gym.local_config import LocalConfiguration
import matplotlib.pyplot as plt

local_config = LocalConfiguration()
local_config.init_config()

def get_value(hp_name, cs, trial):
    hp = cs.get_hyperparameter(hp_name)

    if isinstance(hp, CS.UniformFloatHyperparameter):
        value = float(trial.suggest_float(name=hp_name, low=hp.lower, high=hp.upper, log=hp.log))

    elif isinstance(hp, CS.UniformIntegerHyperparameter):
        value = int(trial.suggest_int(name=hp_name, low=hp.lower, high=hp.upper, log=hp.log))

    elif isinstance(hp, CS.CategoricalHyperparameter):
        hp_type = type(hp.default_value)
        value = hp_type(trial.suggest_categorical(name=hp_name, choices=hp.choices))

    elif isinstance(hp, CS.OrdinalHyperparameter):
        num_vars = len(hp.sequence)
        index = trial.suggest_int(hp_name, low=0, high=num_vars - 1, log=False)
        hp_type = type(hp.default_value)
        value = hp.sequence[index]
        value = hp_type(value)

    elif isinstance(hp, CS.Constant):
        value = hp.value

    else:
        raise ValueError(f"Please implement the support for hps of type {type(hp)}")

    return value

def sample_config_from_optuna(trial, cs):
    config = {}
    for hp_name in cs.get_all_unconditional_hyperparameters():
        value = get_value(hp_name, cs, trial)
        config.update({hp_name: value})

    conditions = cs.get_conditions()
    conditional_hps = list(cs.get_all_conditional_hyperparameters())
    n_conditions = dict(zip(conditional_hps, [len(cs.get_parent_conditions_of(hp)) for hp in conditional_hps]))
    conditional_hps_sorted = sorted(n_conditions, key=n_conditions.get)
    for hp_name in conditional_hps_sorted:
        conditions_to_check = np.where([hp_name in [child.name for child in condition.get_children()] if (isinstance(condition, CS.conditions.AndConjunction) | isinstance(condition, CS.conditions.OrConjunction)) else hp_name == condition.child.name for condition in conditions])[0]
        checks = [conditions[to_check].evaluate(dict(zip([parent.name for parent in conditions[to_check].get_parents()], [config.get(parent.name) for parent in conditions[to_check].get_parents()])) if (isinstance(conditions[to_check], CS.conditions.AndConjunction) | isinstance(conditions[to_check].get_parents())) else {conditions[to_check].parent.name: config.get(conditions[to_check].parent.name)}) for to_check in conditions_to_check]

        if sum(checks) == len(checks):
            value = get_value(hp_name, cs, trial)
            config.update({hp_name: value})

    return config

def objective_mf(trial, bench, opt_space, fidelity_param_id, valid_budgets, targets, prune_thresholds):
    X = sample_config_from_optuna(trial, opt_space)
    value_0, value_1 = 0, 0
    for i in range(len(valid_budgets)):
        X_ = deepcopy(X)
        if "rbv2_" in bench.config.config_id:
            X_.update({"repl": 10})
        X_.update({fidelity_param_id: valid_budgets[i]})
        y = bench.objective_function(X_, logging=True, multithread=False)[0]
        value_0 = y.get(targets[0], value_0)
        value_1 = y.get(targets[1], value_1)
        
        # プルーニングのチェック
        if i < len(prune_thresholds) and y.get(targets[1], float('inf')) > prune_thresholds[i]:
            raise optuna.exceptions.TrialPruned()

    return float(value_0), float(value_1)

def precompute_sh_iters(min_budget, max_budget, eta):
    max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
    return max_SH_iter

def precompute_budgets(max_budget, eta, max_SH_iter, on_integer_scale=False):
    s0 = -np.linspace(start=max_SH_iter - 1, stop=0, num=max_SH_iter)
    budgets = max_budget * np.power(eta, s0)
    if on_integer_scale:
        budgets = budgets.round().astype(int)
    return budgets

def run_optuna(scenario, instance, targets, minimize, on_integer_scale, n_trials, seed, prune_thresholds):
    random.seed(seed)
    np.random.seed(seed)

    bench = benchmark_set.BenchmarkSet(scenario, instance=instance, multithread=False)
    opt_space = bench.get_opt_space(instance)
    opt_space.seed(seed)
    fidelity_space = bench.get_fidelity_space()
    if "rbv2_" in scenario:
        fidelity_param_id = "trainsize"
    else:
        fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
    min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
    max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper
    directions = ["minimize" if m else "maximize" for m in minimize]

    study = optuna.create_study(
        directions=directions,
        sampler=MOTPESampler(seed=seed, n_startup_trials=5),
    )
    reduction_factor = 3
    sh_iters = precompute_sh_iters(min_budget, max_budget, reduction_factor)
    valid_budgets = precompute_budgets(max_budget, reduction_factor, sh_iters, on_integer_scale=on_integer_scale)
    study.optimize(
        func=partial(
            objective_mf,
            bench=bench,
            opt_space=opt_space,
            fidelity_param_id=fidelity_param_id,
            valid_budgets=valid_budgets,
            targets=targets,
            prune_thresholds=prune_thresholds
        ),
        n_trials=n_trials
    )
    time = pd.DataFrame.from_dict([x.get("time") for x in bench.archive])
    X = pd.DataFrame.from_dict([x.get("x") for x in bench.archive])
    Y = pd.DataFrame.from_dict([x.get("y") for x in bench.archive])
    data = pd.concat([time, X, Y], axis=1)
    bench.archive = []

    return data

scenario = 'lcbench'
instance = '167152'
targets = ["val_accuracy", "time"]
minimize = [False, True]
on_integer_scale = False
n_trials = 100
seed = 4

# プルーニングの閾値を設定（時間がこれを超えたらプルーニング）
prune_thresholds = [50, 100, 200, 300]  # プルーニングの閾値

# 多目的最適化を実行
data = run_optuna(scenario, instance, targets, minimize, on_integer_scale, n_trials, seed, prune_thresholds)

# 結果を表示
print(data.columns)
print(data.head())

# 可視化
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data["val_accuracy"], data["time"], c=data["time"], cmap='viridis')
ax.set_xlabel("Validation Accuracy")
ax.set_ylabel("Time")
plt.colorbar(ax.collections[0], label='Time')

plt.show()