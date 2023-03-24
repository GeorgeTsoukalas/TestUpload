import logging
import os
import dsl
import z3

# STUFF I ADDED
from fractions import Fraction
from functools import reduce
import itertools
from itertools import product
import operator
import numpy as np
import math

verbose = False

def lcm(denominators):
    return reduce(lambda a,b: a*b // math.gcd(a,b), denominators)

# this one below seems to be sufficiently general
def numericalInvariant_to_str(parameters, env, type_of_function): #parameters should come in np array format, bias can be in tensor format
    weights = [str(parameters[i]) + "*" + env[i] + " + " for i in range(len(parameters) - 1)]
    weights_str = reduce(operator.concat, weights, "")
    if type_of_function == "affine":
        return weights_str + str(parameters[-1:][0]) + " >= 0"
    elif type_of_function == "equality":
        return weights_str + str(parameters[-1:][0]) + " = 0"
    else:
        assert False # Passed in something which was not a proper function

def floor_ceil_combinations(arr):
    floor_ceil = [math.floor, math.ceil]
    return [list(map(lambda x: x[0](x[1]), zip(combination, arr))) for combination in itertools.product(floor_ceil, repeat=len(arr))]


def smoothed_numerical_invariant_new_nuclear(params): # created 2/13
    weights = (params["weights"][0].detach()).numpy()
    # next we will preprocess the weights to ensure ones that are too close to zero are removed, this is intended to substitute the regularization step 
    for weight_index in range(len(weights)):
        copied_weights = [abs(weight/weights[weight_index]) for weight in weights]
        if max(copied_weights) >= 15: # 15 is a tunable parameter,
            weights[weight_index] = 0
    # print("Weights are ", weights)
    # Find the smallest nonzero element in weights
    smallest_nonzero_weight = 1000
    for weight in weights:
        if abs(weight) < smallest_nonzero_weight and weight != 0.0: # currently negatives count too
            smallest_nonzero_weight = weight
    # the bias range should now be between bias, bias + smallest_nonzero_weight (even if negative)
    bias = float(params["bias"][0].detach())
    bias_new = bias + abs(smallest_nonzero_weight) # I believe, after further analysis, it should always be positive exclusive (but one extra point probably won't hurt at this depth)
    # now do the smoothing operation
    biggest_weight = np.max(np.absolute(weights)) # this does not contain the bias term (it shouldn't), abs is important to not flip sign
    assert biggest_weight != 0 # shouldn't be, but just in case
    new_weights = [weight/biggest_weight for weight in weights]
    # print("New_weights are ", new_weights)
    approximations = []
    N = 5 # this is how precise we want to try for the approximation
    for new_weight in new_weights:
        second_closest_approx_values = (-50, -50)
        second_closest_approx = 999
        closest_approx_values = (0, -100)
        closest_approx = 1000
        if new_weight == 0:
            second_closest_approx_values = (0,1)
            closest_approx_values = (0, 1)
            approximations.append([closest_approx_values, second_closest_approx_values])
            continue
        for i in range(-1*N - 1, N + 1):
            for j in range(1, N+1):
                if (abs(new_weight - i/j) < closest_approx and (np.sign(new_weight) == np.sign(i) or i == 0)):
                    second_closest_approx_values = closest_approx_values
                    second_closest_approx = closest_approx
                    closest_approx = abs(new_weight - i/j)
                    closest_approx_values = (i,j)
        approximations.append([closest_approx_values, second_closest_approx_values])
    # Now we have a choice of two fractional representations 
    #print("approximations are ", approximations)
    smoothed_params = []
    for approximation in itertools.product(*approximations): # this wont ignore if the second closest approx is -1000,-100
        print(approximation)
        new_approximation = [(int(approx[0]/math.gcd(approx[0], approx[1])), int(approx[1]/math.gcd(approx[0], approx[1]))) for approx in approximation]
        least_common_multiple = lcm([frac[1] for frac in new_approximation])
        bias_floor = (bias - abs(smallest_nonzero_weight)) * least_common_multiple/biggest_weight
        bias_ceil = (bias + abs(smallest_nonzero_weight)) * least_common_multiple/biggest_weight
        weights_smoothed = [least_common_multiple * new_approximation[i][0]/new_approximation[i][1] for i in range(len(weights))]
        smoothed_bias_min = int(min(bias_floor, bias_ceil))
        smoothed_bias_max = int(max(bias_floor, bias_ceil))
        for bias_term in range(smoothed_bias_min, smoothed_bias_max + 1): # inclusive
            smoothed_params.append(weights_smoothed + [bias_term])
    #print(smoothed_params)
    #assert False
    return smoothed_params
    # we now have fraction representations, next step is to simplify these down
    #new_approximations = [(int(approx[0]/math.gcd(approx[0], approx[1])), int(approx[1]/math.gcd(approx[0], approx[1]))) for approx in approximations]
    # now we want to multiply each part by the LCM of all the denominators to get smallest integers
    #least_common_multiple = lcm([frac[1] for frac in new_approximations]) # Is this always positive? It isn't! But I don't think it's messed up anything as I enforce denoms to be > 0
    #smoothed_params = []
    #smallest_weight_sign = np.sign(smallest_nonzero_weight)
    #bias_floor = (bias - abs(smallest_nonzero_weight)) * least_common_multiple/biggest_weight
    #bias_ceil = bias_new * least_common_multiple/biggest_weight

    # print("Bias floor is ", bias_floor)
    # print("Bias ceil is ", bias_ceil)
    #bias_new = least_common_multiple * bias/biggest_weight
    #bias_adjusted = least_common_multiple * (bias + smallest_nonzero_weight)/biggest_weight

    #weights_smoothed = [least_common_multiple * new_approximations[i][0]/new_approximations[i][1] for i in range(len(weights))]
    # print("Weights smoothed are ", weights_smoothed)
    #smoothed_bias_min = int(min(bias_floor, bias_ceil))
    #smoothed_bias_max = int(max(bias_floor, bias_ceil))
    # print("Smoothed bias min is ", smoothed_bias_min, " and smoothed_bias max is ", smoothed_bias_max)
    #for bias_term in range(smoothed_bias_min, smoothed_bias_max + 1): # inclusive
    #    smoothed_params.append(weights_smoothed + [bias_term])
        # print("Smoothed parameters are ", smoothed_params)
    # assert False
    #return smoothed_params

def smoothed_numerical_invariant_fourth(params): # the nuclear option
    weights = list((params["weights"][0].detach()).numpy())

    # next we will preprocess the weights to ensure ones that are too close to zero are removed, this is intended to substitute the regularization step 
    for weight_index in range(len(weights)):
        copied_weights = [abs(weight/weights[weight_index]) for weight in weights]
        if max(copied_weights) >= 10: # 10 is a tunable parameter,
            weights[weight_index] = 0
    #print("Weights are ", weights)
    # Find the smallest nonzero element in weights
    smallest_nonzero_weight = 1000
    for weight in weights:
        if abs(weight) < smallest_nonzero_weight and weight != 0.0: # currently negatives count too
            smallest_nonzero_weight = weight
    bias = float(params["bias"][0].detach())
    weights.append(bias)
    assert smallest_nonzero_weight != 0
    assert smallest_nonzero_weight != 1000
    for i in range(len(weights)):
        weights[i]/=abs(smallest_nonzero_weight) # abs is important here! otherwise it flips the sign of the inequality!
    #print(weights)
    
    smoothed_params = floor_ceil_combinations(weights) # this is to test if the weights happen to be slightly off in training
    print("Returns! ", smoothed_params)
    #smoothed_params.append(scaled_weights + [math.floor(bias/smallest_nonzero_weight)])
    #smoothed_params.append(scaled_weights + [math.ceil(bias/smallest_nonzero_weight)])
    return smoothed_params

def smoothed_numerical_invariant_third(params): # these are like the cln2inv ones
    weights = list((params["weights"][0].detach()).numpy())
    bias = float(params["bias"][0].detach())
    weights.append(bias)
    # next we will preprocess the weights to ensure ones that are too close to zero are removed, this is intended to substitute the regularization step 
    for weight_index in range(len(weights)):
        copied_weights = [abs(weight/weights[weight_index]) for weight in weights]
        if max(copied_weights) >= 10: # 10 is a tunable parameter,
            weights[weight_index] = 0
    #print("Weights are ", weights)
    # Find the smallest nonzero element in weights
    smallest_nonzero_weight = 1000
    for weight in weights:
        if abs(weight) < smallest_nonzero_weight and weight != 0.0: # currently negatives count too
            smallest_nonzero_weight = weight
    
    assert smallest_nonzero_weight != 0
    assert smallest_nonzero_weight != 1000
    for i in range(len(weights)):
        weights[i]/=abs(smallest_nonzero_weight) # abs is important!
    print(weights)
    weights_np = np.asarray(weights)
    scaled_weights = np.round(weights_np)
    smoothed_params = [list(scaled_weights)]
    #smoothed_params.append(scaled_weights + [math.floor(bias/smallest_nonzero_weight)])
    #smoothed_params.append(scaled_weights + [math.ceil(bias/smallest_nonzero_weight)])
    return smoothed_params

def smoothed_numerical_invariant_cln2inv(params):
    weights = (params["weights"][0].detach()).numpy()
    # next we will preprocess the weights to ensure ones that are too close to zero are removed, this is intended to substitute the regularization step 
    for weight_index in range(len(weights)):
        copied_weights = [abs(weight/weights[weight_index]) for weight in weights]
        if max(copied_weights) >= 10: # 15 is a tunable parameter,
            weights[weight_index] = 0
    # print("Weights are ", weights)
    # Find the smallest nonzero element in weights
    smallest_nonzero_weight = 1000
    for weight in weights:
        if abs(weight) < smallest_nonzero_weight and weight != 0.0: # currently negatives count too
            smallest_nonzero_weight = weight
    
    assert smallest_nonzero_weight != 0
    assert smallest_nonzero_weight != 1000
    bias = float(params["bias"][0].detach())
    bias_floor = (bias/abs(smallest_nonzero_weight) - 1)
    bias_ceil = (bias/abs(smallest_nonzero_weight) + 1)
    weights /= abs(smallest_nonzero_weight) # again this abs is super important!
    print(weights)
    #print(weights)
    max_denominator = 5 # tunable
    frac_approximations = []
    denominator = 1
    for coeff in weights:
        frac = Fraction.from_float(float(coeff)).limit_denominator(max_denominator)
        frac_approximations.append(frac)
        denominator = denominator * frac.denominator // math.gcd(denominator, frac.denominator) # this is essentially a fold, in the style of cln2inv code
    new_weights = [math.floor(a * denominator) for a in frac_approximations] # no bias mobility yet

    smoothed_params = []
    smoothed_bias_min = int(min(bias_floor, bias_ceil))
    smoothed_bias_max = int(max(bias_floor, bias_ceil))
    for bias_term in range(smoothed_bias_min, smoothed_bias_max + 1):
        smoothed_params.append(new_weights + [bias_term])
    #print("smoothed params are ", smoothed_params)
    #assert False #testing
    return smoothed_params



# Looks like I had already made this one sufficiently general
def smoothed_numerical_invariant_new(params): # created 2/2
    weights = (params["weights"][0].detach().cpu()).numpy() # added .cpu() 3/23 to support GPU
    # next we will preprocess the weights to ensure ones that are too close to zero are removed, this is intended to substitute the regularization step 
    for weight_index in range(len(weights)):
        copied_weights = [abs(weight/weights[weight_index]) for weight in weights]
        if max(copied_weights) >= 15: # 15 is a tunable parameter,
            weights[weight_index] = 0
    # print("Weights are ", weights)
    # Find the smallest nonzero element in weights
    smallest_nonzero_weight = 1000
    for weight in weights:
        if abs(weight) < smallest_nonzero_weight and weight != 0.0: # currently negatives count too
            smallest_nonzero_weight = weight
    # the bias range should now be between bias, bias + smallest_nonzero_weight (even if negative)
    bias = float(params["bias"][0].detach())
    bias_new = bias + abs(smallest_nonzero_weight) # I believe, after further analysis, it should always be positive exclusive (but one extra point probably won't hurt at this depth)
    # now do the smoothing operation
    biggest_weight = np.max(np.absolute(weights)) # this does not contain the bias term (it shouldn't)
    assert biggest_weight != 0 # shouldn't be, but just in case
    new_weights = [weight/biggest_weight for weight in weights] # TODO: check if this should be abs (but I don't know if all negative happens in practice)
    # print("New_weights are ", new_weights)
    approximations = []
    N = 4 # this is how precise we want to try for the approximation
    for new_weight in new_weights:
        closest_approx_values = (-100, -100)
        closest_approx = 1000
        if new_weight == 0:
            closest_approx_values = (0, 1)
            approximations.append(closest_approx_values)
            continue
        for i in range(-1*N - 1, N + 1):
            for j in range(1, N+1):
                if (abs(new_weight - i/j) < closest_approx and (np.sign(new_weight) == np.sign(i) or i == 0)):
                    closest_approx = abs(new_weight - i/j)
                    closest_approx_values = (i,j)
        approximations.append(closest_approx_values)
    # we now have fraction representations, next step is to simplify these down
    new_approximations = [(int(approx[0]/math.gcd(approx[0], approx[1])), int(approx[1]/math.gcd(approx[0], approx[1]))) for approx in approximations]
    # now we want to multiply each part by the LCM of all the denominators to get smallest integers
    least_common_multiple = lcm([frac[1] for frac in new_approximations]) # Is this always positive? It isn't! But I don't think it's messed up anything as I enforce denoms to be > 0
    smoothed_params = []
    #smallest_weight_sign = np.sign(smallest_nonzero_weight)
    bias_floor = (bias - abs(smallest_nonzero_weight)) * least_common_multiple/biggest_weight
    bias_ceil = bias_new * least_common_multiple/biggest_weight

    # print("Bias floor is ", bias_floor)
    # print("Bias ceil is ", bias_ceil)
    #bias_new = least_common_multiple * bias/biggest_weight
    #bias_adjusted = least_common_multiple * (bias + smallest_nonzero_weight)/biggest_weight

    weights_smoothed = [least_common_multiple * new_approximations[i][0]/new_approximations[i][1] for i in range(len(weights))]
    # print("Weights smoothed are ", weights_smoothed)
    smoothed_bias_min = int(min(bias_floor, bias_ceil))
    smoothed_bias_max = int(max(bias_floor, bias_ceil))
    # print("Smoothed bias min is ", smoothed_bias_min, " and smoothed_bias max is ", smoothed_bias_max)
    for bias_term in range(smoothed_bias_min-1, smoothed_bias_max + 2): # inclusive
        smoothed_params.append(weights_smoothed + [bias_term])
        # print("Smoothed parameters are ", smoothed_params)
    # assert False
    return smoothed_params


def smoothed_numerical_invariant(params):
    weights = (params["weights"][0].detach()).numpy()
    biggest_weight =  abs(np.max(weights)) 
    assert biggest_weight != 0 
    bias = float(params["bias"][0].detach())/biggest_weight
    new_weights = [weight/biggest_weight for weight in weights]
    approximations = []
    N = 5 # this is how precise we want to try for the approximation
    for new_weight in new_weights:
        closest_approx_values = (-100, -100)
        closest_approx = 1000
        for i in range(-5, N+1):
            for j in range(1, N+1):
                if (abs(new_weight - i/j) < closest_approx and (np.sign(new_weight) == np.sign(i) or i == 0)):
                    closest_approx = abs(new_weight - i/j)
                    closest_approx_values = (i,j)
        approximations.append(closest_approx_values)
    # Problem which the following code addresses: (5,5) = (1,1) as an approximation but the latter one will be chosen. We want the approximation denominators to be small as possible.
    new_approximations = [(int(approx[0]/math.gcd(approx[0], approx[1])), int(approx[1]/math.gcd(approx[0], approx[1]))) for approx in approximations]
    least_common_multiple = lcm([frac[1] for frac in new_approximations])
    #print("Least common multiple becomes (pls dont be negative)", least_common_multiple)
    return [least_common_multiple * approximations[i][0]/approximations[i][1] for i in range(len(weights))] + [ math.ceil(least_common_multiple * bias)] # TODO: work on locating better bias term.

# This one is also sufficiently general
def print_program2(program, env, smoothed = False):
    if program.name == "affine" or program.name == "equality":
        if smoothed:
            weights = smoothed_numerical_invariant(program.parameters)
            print("( " + program.name + " " + numericalInvariant_to_str(weights, env, program.name))
        else:
            weights = list((program.parameters["weights"][0].detach().cpu()).numpy())
            bias = float(program.parameters["bias"][0].detach())
            weights.append(bias) #converting to proper form
            print("( " + program.name + " " + numericalInvariant_to_str(weights, env, program.name))
    else:
        print("(" + program.name)
        for submodule, function in program.submodules.items():
            print_program2(function, env, smoothed)
        print(" )")

def invariant_from_program_new(program, params, env):
    if program.name == "affine":
        weights = params.pop(0) # again, the question is does this modify params?
        z3_ineq = 0
        z3_ineq = sum(weight * z3.Real(var) for weight, var in zip(weights, env)) + weights[-1] >= 0
        return z3_ineq
    elif program.name == "equality":
        weights = params.pop(0) # again, the question is does this modify params?
        z3_eq = 0
        z3_eq = sum(weight * z3.Real(var) for weight, var in zip(weights, env)) + weights[-1] == 0
        return z3_eq
    elif program.name == "and":
        funcs = []
        for submodule, function in program.submodules.items():
            funcs.append(function) # what's the real point of this?
        return z3.And(invariant_from_program_new(funcs[0], params, env), invariant_from_program_new(funcs[1], params, env))
    elif program.name == "or":
        funcs = []
        for submodule, function in program.submodules.items():
            funcs.append(function) # what's the real point of this?
        return z3.Or(invariant_from_program_new(funcs[0], params, env), invariant_from_program_new(funcs[1], params, env))
        
# This one I've adjusted to be more general
def invariant_from_program(program, env):
    # assume vars are x, y
    if program.name == "affine":
        weights = smoothed_numerical_invariant(program.parameters)
        z3_ineq = 0
        # TODO: make it more general, look at line 402 from cln_training.py
        #z3_ineq = weights[0] * z3.Real(env[0]) + weights[1] * z3.Real(env[1]) + weights[2] * 1 >= 0.0 
        z3_ineq = sum(weight * z3.Real(var) for weight, var in zip(weights, env)) + weights[-1] >= 0.0
        return z3_ineq
    elif program.name == "equality":
        print("The parameters are ", program.parameters)
        weights = smoothed_numerical_invariant(program.parameters)
        print(" And the weights are ", weights)
        z3_eq = 0
        # TODO: make it more general according to line 402 from cln_training.py
        #z3_eq = weights[0] * z3.Real(env[0]) + weights[1] * z3.Real(env[1]) + weights[2] * 1 == 0.0 # double equals for defining the equality
        z3_eq = sum(weight*z3.Real(var) for weight, var in zip(weights, env)) + weights[-1] == 0.0
        return z3_eq
    elif program.name == "and":
        funcs = []
        for submodule, function in program.submodules.items():
            funcs.append(function)
        return z3.And(invariant_from_program(funcs[0], env), invariant_from_program(funcs[1], env))
    elif program.name == "or":
        funcs = []
        for submodule, function in program.submodules.items():
            funcs.append(function)
        return z3.Or(invariant_from_program(funcs[0], env), invariant_from_program(funcs[1], env))
    # let's see if it actually works like this  
def init_logging(save_path):
    logfile = os.path.join(save_path, 'log.txt')

    # clear log file
    with open(logfile, 'w'):
        pass
    # remove previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')

def log_and_print(line):
    if verbose:
        print(line)
    logging.info(line)

def print_program(program, ignore_constants=True):
    if not isinstance(program, dsl.LibraryFunction):
        return program.name
    else:
        collected_names = []
        for submodule, functionclass in program.submodules.items():
            collected_names.append(print_program(functionclass, ignore_constants=ignore_constants))
        if program.has_params:
            parameters = "params: {}".format(program.parameters.values())
            if not ignore_constants:
                collected_names.append(parameters)
        joined_names = ', '.join(collected_names)
        return program.name + "(" + joined_names + ")"

def print_program_dict(prog_dict):
    log_and_print(print_program(prog_dict["program"], ignore_constants=True))
    log_and_print("struct_cost {:.4f} | score {:.4f} | path_cost {:.4f} | time {:.4f}".format(
        prog_dict["struct_cost"], prog_dict["score"], prog_dict["path_cost"], prog_dict["time"]))
