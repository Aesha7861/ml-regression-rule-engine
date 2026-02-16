import argparse
import numpy as np
import pandas as pd
import operator 

def framework(pairs, arr):
    """
    Args:
       - pairs:  a list of (cond, calc) tuples. calc() must be an executable
       - arr: a numpy array with the features in order feat_1, feat_2, ...
    
    Executes the first calc() whose cond returns True.
    Returns None if no condition matches.
    """
    targets = []

    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
        
    return targets


def cond_eval(condition, arr):
    """evaluate a condition
        - condition: must be a tupe of (int, string, float). The second entry must be a string from the list below, describing the operator. Third entry of the tuple must be a float). If condition is None, it is always evaluated to true.
        - arr: array on which the condition is evaluated

    The python operator package is used. Second entry in condition must be one of those:
       ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    """
    ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition is None:
        return True
    
    op = ops[condition[1]]
    return op(arr[condition[0]], condition[2])


def main(args):
    # Read CSV with column names so can map feat_* -> correct index robustly
    df = pd.read_csv(args.eval_file_path)
    data_array = df.values

    # Robust column->index mapping (avoids any feat_0 vs feat_1 confusion)
    idx_switch = df.columns.get_loc("feat_103")
    idx_f42    = df.columns.get_loc("feat_42")
    idx_f19    = df.columns.get_loc("feat_19")
    idx_f109   = df.columns.get_loc("feat_109")

    # Derived thresholds (switch feature feat_103)
    t1 = 0.19994787871837616
    t2 = 0.4999445378780365
    t3 = 0.6999787092208862

    # Region formulas (piecewise linear)
    def calc_r0(arr):
        return (1.25 * arr[idx_f42]) + (-0.75 * arr[idx_f19]) + (-0.25 * arr[idx_f109])

    def calc_r1(arr):
        return (-2.05 * arr[idx_f42]) + (0.85 * arr[idx_f19]) + (-1.35 * arr[idx_f109])

    def calc_r2(arr):
        return (-0.65 * arr[idx_f42]) + (0.85 * arr[idx_f19]) + (-0.55 * arr[idx_f109])

    def calc_r3(arr):
        return (1.35 * arr[idx_f42]) + (1.25 * arr[idx_f19]) + (-1.45 * arr[idx_f109])

    # Conditions: evaluated in order; first match wins
    condition0 = (idx_switch, "<=", float(t1))
    condition1 = (idx_switch, "<=", float(t2))
    condition2 = (idx_switch, "<=", float(t3))
    condition3 = None  # else

    pair_list = [
        (condition0, calc_r0),
        (condition1, calc_r1),
        (condition2, calc_r2),
        (condition3, calc_r3),
    ]

    preds = framework(pair_list, data_array)

    return preds

    
def main_example(args):

    # Example: 
    test_arr = np.ones((10,10))

    def calc1(arr):
        """square first array column"""
        return arr[0]**2

    def calc2(arr):
        """add columns 3 and 4"""
        return arr[2] + arr[3]

    condition1 = (0,">=", 0.5)
    condition2 = (8, "==", 0.0)

    predict_targets = framework([(condition1, calc1), (condition2, calc2)], test_arr)
    print (predict_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Task 2")
    parser.add_argument("--eval_file_path", required=True, help="Path to EVAL_<ID>.csv")
    args = parser.parse_args()

    target02 = main(args)
