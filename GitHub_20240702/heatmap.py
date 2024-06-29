from yahpo_gym import benchmark_set
from yahpo_gym.benchmarks import *
import ConfigSpace as CS
import random
import pandas as pd
import numpy as np
from itertools import product
from copy import deepcopy
from yahpo_gym.local_config import LocalConfiguration
import seaborn as sns
import matplotlib.pyplot as plt

# Local configuration initialization
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
        checks = [conditions[to_check].evaluate(dict(zip([parent.name for parent in conditions[to_check].get_parents()], [config.get(parent.name) for parent in conditions[to_check].get_parents()])) if (isinstance(conditions[to_check], CS.conditions.AndConjunction) | isinstance(conditions[to_check], CS.conditions.OrConjunction)) else {conditions[to_check].parent.name: config.get(conditions[to_check].parent.name)}) for to_check in conditions_to_check]

        if sum(checks) == len(checks):
            value = get_value(hp_name, cs, trial)
            config.update({hp_name: value})

    return config

def objective_mf(trial, bench, opt_space, fidelity_param_id, valid_budgets, target):
    X = sample_config_from_optuna(trial, opt_space)

    results = []
    for i in range(len(valid_budgets)):
        X_ = deepcopy(X)
        if "rbv2_" in bench.config.config_id:
            X_.update({"repl":10})  # manual fix required for rbv2_
        X_.update({fidelity_param_id: valid_budgets[i]})
        y = bench.objective_function(X_, logging=True, multithread=False)[0]
        results.append({**X_, fidelity_param_id: valid_budgets[i], target: y.get(target)})

        if trial.should_prune():
            raise optuna.TrialPruned()

    return results

def precompute_sh_iters(min_budget, max_budget, eta):
    max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
    return max_SH_iter

def precompute_budgets(max_budget, eta, max_SH_iter, on_integer_scale=False):
    s0 = -np.linspace(start=max_SH_iter - 1, stop=0, num=max_SH_iter)
    budgets = max_budget * np.power(eta, s0)
    if on_integer_scale:
        budgets = budgets.round().astype(int)
    return budgets

def generate_evenly_spaced_parameters(cs, num_splits):
    param_grid = {}

    for hp_name in cs.get_all_unconditional_hyperparameters():
        hp = cs.get_hyperparameter(hp_name)

        if isinstance(hp, CS.UniformFloatHyperparameter):
            values = np.linspace(hp.lower, hp.upper, num_splits).tolist()
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            values = np.linspace(hp.lower, hp.upper, num_splits, dtype=int).tolist()
        elif isinstance(hp, CS.CategoricalHyperparameter):
            values = hp.choices
        elif isinstance(hp, CS.OrdinalHyperparameter):
            values = hp.sequence
        elif isinstance(hp, CS.Constant):
            values = [hp.value]
        else:
            raise ValueError(f"Please implement the support for hps of type {type(hp)}")
        
        param_grid[hp_name] = values

    return param_grid

def generate_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    return all_combinations

def run_grid_search(scenario, instance, target, minimize, on_integer_scale, param_combinations):
    results = []
    bench = benchmark_set.BenchmarkSet(scenario, instance=instance, multithread=False)
    opt_space = bench.get_opt_space(instance)
    fidelity_space = bench.get_fidelity_space()
    if "rbv2_" in scenario:
        fidelity_param_id = "trainsize"
    else:
        fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
    max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper
    
    for params in param_combinations:
        params.update({fidelity_param_id: max_budget})
        if "rbv2_" in bench.config.config_id:
            params.update({"repl": 10})
        
        y = bench.objective_function(params, logging=True, multithread=False)[0]
        results.append({**params, target: y.get(target)})

    return pd.DataFrame(results)

scenario = 'lcbench'
instance = '167152'
target = "val_accuracy"
minimize = False
on_integer_scale = False
num_splits = 5
seed = 4

random.seed(seed)
np.random.seed(seed)

# パラメータグリッドとハイパーパラメータ組み合わせの生成
bench = benchmark_set.BenchmarkSet(scenario, instance=instance, multithread=False)
opt_space = bench.get_opt_space(instance)
param_grid = generate_evenly_spaced_parameters(opt_space, num_splits)
param_combinations = generate_combinations(param_grid)

# グリッドサーチを実行し，データ収集
data = run_grid_search(scenario, instance, target, minimize, on_integer_scale, param_combinations)

# 結果を表示
print(data)

# ヒートマップを作成するためのハイパーパラメータを選択
# 種類：batch_size, epoch, learning_rate, max_dropout, max_units, momentum, num_layers
hp1 = 'learning_rate'
hp2 = 'batch_size'

# epochが52のデータだけを抽出
data_epoch_52 = data[data['epoch'] == 52]

# ヒートマップ用のデータを準備
heatmap_data = data_epoch_52.pivot_table(index=hp1, columns=hp2, values=target)

# ヒートマップを作成
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='viridis')
plt.title('Heatmap of ' + target + ' for epoch 52')
plt.show()

# CSVファイルに保存
data.to_csv('grid_search_results5.csv', index=False)

