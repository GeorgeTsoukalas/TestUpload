import z3
import random

class InvariantChecker():
    
    def __init__(self, c_file, smt_check_path):
        self.c_file = c_file
        
        # read top pre rec post checks in
        with open(smt_check_path+'/'+c_file+'.smt.1') as f:
            self.top = f.read()
            
        with open(smt_check_path+'/'+c_file+'.smt.2') as f:
            self.pre = f.read()
            
        with open(smt_check_path+'/'+c_file+'.smt.3') as f:
            self.loop = f.read()
            
        with open(smt_check_path+'/'+c_file+'.smt.4') as f:
            self.post = f.read()
            
        # self.solver = z3.Solver()
        # self.solver.set("timeout", 2000)
        
    def check(self, inv_str, env):
        for ind in range(3):
            check = [self.pre, self.post, self.loop][ind] # The order here may actually matter, if we want more implication examples or less, I think we want less
            full_check = self.top + inv_str + check
            solver = z3.Solver()
            solver.set("timeout", 2000)
            solver.from_string(full_check)
            res = solver.check()
            current_part = ""
            if ind == 0:
                current_part = "pre"
                #print(self.post)
            elif ind == 1:
                current_part = "post"
                #print(self.pre)
            else:
                current_part = "loop"
            print("The result is ", res, " during the ", current_part)

            #print(solver.model())
            # self.solver.push()
            # self.solver.from_string(full_check)
            # res = self.solver.check()
            # self.solver.pop()
               
            if res != z3.unsat and current_part == "loop":
                print("Not a valid invariant: here is a counterexample")
                model = solver.model()
                print(model)
                z3_env = list(map(lambda x: z3.Int(x), env))
                #print(list(z3_env))
                #assert False # Testing
                z3_env_new = list(map(lambda x: z3.Int(x), map(lambda x: x + '!', env)))
                print(z3_env)
                print(z3_env_new)
                # testing to see how z3 works when a variable isn't there
                print(env)
                for var in z3_env:
                    print(var, " ", model[var])
                returned_assignment = []
                returned_assignment_new = []
                for var in z3_env:
                    if model[var] != None:
                        returned_assignment.append(model[var].as_long())
                    else:
                        # Ideally, choose a new data point for z in the range of the data seen before, but here the data is not passed so I think it is alright to just pick a random integer from [0, 10]
                        random_choice = random.randint(0, 10)
                        returned_assignment.append(random_choice)     
                for var in z3_env_new:
                    if model[var] != None:
                        returned_assignment_new.append(model[var].as_long())
                    else:
                        # Ideally, choose a new data point for z in the range of the data seen before, but here the data is not passed so I think it is alright to just pick a random integer from [0, 10]
                        random_choice = random.randint(0, 10)
                        returned_assignment_new.append(random_choice)             
                return False, current_part, [returned_assignment, returned_assignment_new] #[[model[var].as_long() for var in z3_env], [model[var].as_long() for var in z3_env_new]]#[[model[i].as_long(), model[j].as_long()], [model[i_new].as_long(), model[j_new].as_long()]]
            elif res != z3.unsat: # pre or post, only return the single point
                print("Not a valid invariant: here is a counterexample")
                model = solver.model()
                z3_env = list(map(lambda x: z3.Int(x), env))
                #print(list(z3_env))
                #assert False # Testing
                print(model)
                print(env)
                # testing to see how z3 works when a variable isn't there
                for var in z3_env:
                    print(var, " ", model[var])
                #print([model[var].as_long() for var in z3_env])
                returned_assignment = [] # doing this in a longer format to make changing the random assignment easier
                for var in z3_env:
                    if model[var] != None:
                        returned_assignment.append(model[var].as_long())
                    else:
                        # Ideally, choose a new data point for z in the range of the data seen before, but here the data is not passed so I think it is alright to just pick a random integer from [0, 10]
                        random_choice = random.randint(0, 10)
                        returned_assignment.append(random_choice)
                print("The returned assignment is ", returned_assignment)
                #print("Did not satisfy", check)
                return False, current_part, returned_assignment #[model[var].as_long() for var in z3_env] #[model[i].as_long(), model[j].as_long()]
        return True, ""

    def check_cln(self, inv_smts, env):
        #print("Inv_smts are", inv_smts)
        correct = False
        vals = []
        for inv_smt in inv_smts:
            #print("Inv_str is ", inv_smt)
            inv_str = inv_smt.sexpr()
            print("Inv_str after sexpr is ", inv_str)
            inv_str = inv_str.replace('|', '')
            print("Inv_str after | replace is ", inv_str)
            correct = self.check(inv_str, env)
            if correct[0]:
                return True, inv_str, []
            else:
                vals = correct[2]
        return False, '', correct[1], vals # correct[1] is the part of the verification that failed

            
