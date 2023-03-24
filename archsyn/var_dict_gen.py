import subprocess
dictionary = {}
src_path = 'benchmarks-cln2inv/code2inv/c/'
for problem_num in range(134):
    # Run the ctags command to generate a tags file
    file_name = src_path + str(problem_num) + ".c"
    subprocess.run(["ctags", "-R", "-x", "-f", "C_tags", file_name], capture_output=True)

    # Parse the tags file to extract the variable names
    with open("C_tags") as f:
        lines = f.readlines()
        variables = [line.split("\t")[0] for line in lines if "variable" in line]
        print(variables)
    dictionary[problem_num] = variables
print(dictionary)

var_types = {
    1: ["x", "y"],
    2: ["n", "x", "y"],
    3: ["n", "x"]
}
dictionary = {
    99: ["n", "x", "y"],
    98: ["i", "j", "x", "y"],
    97: ["i", "j", "x", "y"],
    96: ["i", "j", "x", "y"],
    95: ["i", "j", "x", "y"],
    94: ["i", "j", "k", "n"],
    93: ["i", "n", "x", "y"],
    92: ["x", "y", "z1", "z2", "z3"],
    91: ["x", "y"],
    90:  ["lock", "x", "y", "v1", "v2", "v3"],
    9: ["x", "y"],
    89: ["lock", "x", "y", "v1", "v2", "v3"],
    88: ["lock", "x", "y"],
    87: ["lock", "x", "y"],
    86: ["x", "y", "z1", "z2", "z3"],
    85: ["x", "y", "z1", "z2", "z3"],
    84: ["x", "y"],
    83: ["x", "y"],
    82:["i", "x", "y", "z1", "z2", "z3"],
    81:["i", "x", "y", "z1", "z2", "z3"],
    80: ["i", "x", "y", "z1", "z2", "z3"],
    8: ["x", "y"],
    79: ["i", "x", "y"],
    78: ["i", "x", "y"],
    77: ["i", "x", "y"],
    76: ["c", "y", "z", "x1", "x2", "x3"],
    75: ["c", "y", "z", "x1", "x2", "x3"],
    74: ["c", "y", "z", "x1", "x2", "x3"],
    73: ["c", "y", "z"],
    72: ["c", "y", "z"],
    71: ["c", "y", "z"],
    70: ["x", "y"],
    7:  ["x", "y"],
    69: ["n", "v1", "v2", "v3", "x", "y"],
    68: ["n", "y", "x"],
    67: ["n", "y", "x"],
    66: ["x", "y"],
    65: ["x", "y"],
    64: ["x", "y"],
    63: ["x", "y"],
    62: ["c", "n", "v1", "v2", "v3"],
    61: ["c", "n", "v1", "v2", "v3"],
    60: ["c", "n", "v1", "v2", "v3"],
    6: ["v1", "v2", "v3", "x", "size", "y", "z"] # like in other cases, some of these variables are predeclared to be a specific value. It is worth checking to see if we can reduce the variable load by looking at the interaction with z3
    59:["c", "n", "v1", "v2", "v3"],
    58:["c", "n", "v1", "v2", "v3"],
    57:["c", "n", "v1", "v2", "v3"],
    56:["c", "n", "v1", "v2", "v3"],
    55:["c", "n", "v1", "v2", "v3"],
    54:["c", "n", "v1", "v2", "v3"],
    53: ["c", "n", "v1", "v2", "v3"],
    52: ["c"],
    51: ["c"],
    50: ["c"],
    5: ["x", "size", "y", "z"],
    49:["c", "n"],
    48:["c", "n"],
    47:["c", "n"],
    46:["c", "n"],
    45:["c", "n"],
    44:["c", "n"],
    43:["c", "n"],
    42:["c", "n"],
    41:["c", "n"],
    40: ["c", "n"],
    4: ["x", "y", "z"],
    39:["n", "c"],
    38:["n", "c"],
    37:["c"],
    36:["c"],
    35:["c"],
    34:["n", "v1", "v2", "v3", "x"],
    33:["n", "v1", "v2", "v3", "x"],
    32:["n", "v1", "v2", "v3", "x"],
    31: ["n", "v1", "v2", "v3", "x"],
    30: ["x"],
    3: ["x", "y", "z"],
    29:["n", "x"],
    28:["n", "x"],
    27:["n", "x"],
    26: ["n", "x"],
    25: ["x"],
    24:["i", "j"],
    23: ["i", "j"],
    22: ["x", "m", "n", "z1", "z2", "z3"],
    21: ["x", "m", "n", "z1", "z2", "z3"],
    20:  ["x", "m", "n", "z1", "z2", "z3"],
    2: ["x", "y"],
    19: ["x", "m", "n", "z1", "z2", "z3"],
    18:["x", "m", "n"],
    17:["x", "m", "n"],
    16: ["x", "m", "n"],
    15: ["x", "m", "n"],
    14:["x", "y", "z1", "z2", "z3"],
    133: ["n", "x"],
    132: ["i", "j", "c", "t"],
    131: ["d1", "d2", "d3", "x1", "x2", "x3"],
    130: ["d1", "d2", "d3", "x1", "x2", "x3"],
    13: ["x", "y", "z1", "z2", "z3"],
    129: ["x", "y", "z1", "z2", "z3"],
    128: ["x", "y"],
    127: ["i", "j", "x", "y", "z1", "z2", "z3"],
    126: ["i", "j", "x", "y", "z1", "z2", "z3"],
    125: ["i", "j", "x", "y"],
    124: ["i", "j", "x", "y"],
    123: ["i", "size", "sn", "v1", "v2", "v3"],
    122: ["i", "size", "sn", "v1", "v2", "v3"],
    121: ["i", "sn"],
    120: ["i", "sn"],
    12: ["x", "y", "z1", "z2", "z3"],
    119: ["i", "size", "sn"],
    118: ["i", "size", "sn"],
    117: ["sn", "v1", "v2", "v3", "x"],
    116: ["sn", "v1", "v2", "v3", "x"],
    115: ["sn", "x"],
    114: ["sn", "x"],
    113: ["i", "n", "sn", "v1", "v2", "v3"],
    112: ["i", "n", "sn", "v1", "v2", "v3"],
    111: ["i", "n", "sn"],
    110: ["i", "n", "sn"],
    1: ["x", "y"],
    10: ["x", "y"],
    100: ["n", "x", "y"],
    101: ["n", "x"],
    102: ["n", "x"],
    103: ["x"],
    104: ["n", "v1", "v2", "v3", "x"],
    105: ["n", "v1", "v2", "v3", "x"],
    106: ["a", "m", "j", "k"],
    107: ["a", "m", "j", "k"],
    108: ["a","c", "m", "j", "k"],
    109: ["a","c", "m", "j", "k"],
    11: ["x", "y", "z1", "z2", "z3"]

}