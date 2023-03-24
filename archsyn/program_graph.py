import copy
import dsl

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 

from utils.data_loader import pad_minibatch, unpad_minibatch, flatten_tensor
from utils.logging import print_program, log_and_print
from dsl.library_functions import lib_funcions, VarSelFunction, AffineFunc, LogicAndFunction, LogicOrFunction

early_clear = True
arch_search = True

# edge in graph (connect program with next node)
class Edge(nn.Module):
    def __init__(self, from_node, to_node, type_sign, device, randset=0):
        super(Edge, self).__init__()

        self.from_node = from_node
        self.to_node = to_node
        self.type_sign = type_sign
        self.W, self.W_id = self.init_w(device)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device
        self.randset = randset

    def init_w(self, device):
        weights = torch.zeros(len(self.to_node.prog_dict[self.type_sign]),
                        requires_grad = False,
                        dtype = torch.float,
                        device = device)
        weights.requires_grad = True
        weight_ids = torch.ones(len(self.to_node.prog_dict[self.type_sign]),
                                requires_grad = False,
                                dtype=torch.long,
                                device=device)
        return weights, weight_ids

    def get_params(self):
        return self.W

    def get_next_node(self):
        return self.to_node

    def reduce_weights(self, topN, penalty=0.0):

        if torch.sum(self.W_id) == 0:
            print('valid weight should not be zero')
            exit(0)

        with torch.no_grad():
            # get structure cost
            valid_pos = [p_id.item() for p_id in torch.where(self.W_id==1)[0]]
            prog_cost = self.to_node.get_program_cost(self.type_sign, valid_pos)
            prog_cost = torch.tensor(prog_cost).to(self.device)
            assert prog_cost.shape[0] == self.W[self.W_id==1].shape[0]
            # select topN
            topN = min([topN, torch.sum(self.W_id).item()])
            if penalty == 0:
                new_W, top_idxs = torch.topk(self.W[self.W_id==1], topN)
            else:
                new_W, top_idxs = torch.topk(self.softmax(self.W[self.W_id==1])-penalty*prog_cost, topN)
            self.W = new_W
            sel_idxs = torch.where(self.W_id==1)[0][top_idxs]

        # grad
        self.W.requires_grad = True
        self.W_id = torch.ones_like(self.W, dtype=torch.long, requires_grad=False)

        return sel_idxs

    def select_topN_weights(self, selTopN, sorted=True, get_prob=False, penalty=0.0):
        if torch.sum(self.W_id) == 0:
            print('valid weight should not be zero')
            exit(0)

        with torch.no_grad():
            selTopN = min([selTopN, torch.sum(self.W_id).item()-1])
            valid_W = self.W[self.W_id==1]
            if get_prob:
                valid_prob = self.softmax(valid_W)
            if sorted:
                # get structure cost
                valid_pos = [p_id.item() for p_id in torch.where(self.W_id==1)[0]]
                prog_cost = self.to_node.get_program_cost(self.type_sign, valid_pos)
                prog_cost = torch.tensor(prog_cost).to(self.device)
                assert prog_cost.shape[0] == valid_W.shape[0]
                # get topN
                if penalty == 0:
                    _, sort_ids = torch.sort(valid_W, descending=True)
                else:
                    _, sort_ids = torch.sort(self.softmax(valid_W)-penalty*prog_cost, descending=True)
                sel_id = sort_ids[selTopN]
            else:
                sel_id = selTopN
            self.W = valid_W[sel_id:sel_id+1]
            if get_prob:
                return_prob = valid_prob[sel_id]

            # debug double check
            return_id = torch.where(self.W_id==1)[0][sel_id:sel_id+1]

        # grad
        self.W.requires_grad = True
        self.W_id = torch.ones_like(self.W, dtype=torch.long, requires_grad=False)

        if get_prob:
            return return_id, return_prob
        return return_id

    def select_valid_weights(self):
        if torch.sum(self.W_id) == 0:
            print('valid weight should not be zero')
            exit(0)
        # new weight
        with torch.no_grad():
            valid_W = self.W[self.W_id==1]
            valid_ids = torch.where(self.W_id==1)[0]

            self.W = valid_W
        # grad
        self.W.requires_grad = True
        self.W_id = torch.ones_like(self.W, dtype=torch.long, requires_grad=False)

        return valid_ids

    def get_valid_pos(self):
        return torch.where(self.W_id==1)[0]

    # TODO: need to reconsider, whether w_id should be added
    def delete_weights(self, del_ids):
        # get new
        del_ids = np.sort(del_ids)
        with torch.no_grad():
            new_W = [self.W[:del_ids[0]]]
            new_W_id = [self.W_id[:del_ids[0]]]   # for debug
            if len(del_ids) != 1:
                new_W = new_W + [self.W[cur_id+1:next_id] for cur_id, next_id in zip(del_ids[:-1], del_ids[1:])]
                new_W_id = new_W_id + [self.W_id[cur_id+1:next_id] for cur_id, next_id in zip(del_ids[:-1], del_ids[1:])]   # for debug
            new_W = new_W + [self.W[del_ids[-1]+1:]]
            new_W_id = new_W_id + [self.W_id[del_ids[-1]+1:]]   # for debug
            if torch.sum(torch.cat(new_W_id)) == 0:     # for debug
                pdb.set_trace()

            self.W = torch.cat(new_W)
            self.W_id = torch.ones_like(self.W, dtype=torch.long, requires_grad=False)
        # grad
        self.W.requires_grad = True

    # when weight and node are not same
    def select_weights(self, del_ids):
        # get new
        del_ids = np.sort(del_ids)
        with torch.no_grad():
            self.W_id[del_ids] = 0

    # add zero to edge if node contain more unrelated program
    def pad_weights(self, total_num, head=True, select_idxs=None):
        # select certain weights
        if select_idxs is not None:
            assert len(select_idxs) == self.W.shape[0]
            new_weights = torch.zeros(total_num, requires_grad = False,
                                dtype = torch.float,
                                device = self.device)
            new_w_id = torch.zeros(total_num, requires_grad = False,
                                dtype = torch.long,
                                device = self.device)
            with torch.no_grad():
                new_weights[select_idxs] = self.W
                new_w_id[select_idxs] = 1
                self.W = new_weights
                self.W.requires_grad = True
                self.W_id = new_w_id
        # pad
        else:
            pad_num = total_num - self.W.shape[0]
            padded_w_id = torch.zeros(pad_num, dtype=torch.long, requires_grad=False, device=self.device)
            padded_w = torch.zeros(pad_num, dtype=torch.float, requires_grad=True, device=self.device)
            with torch.no_grad():
                if head:
                    self.W_id = torch.cat([padded_w_id, self.W_id])
                    self.W = torch.cat([padded_w, self.W])
                    self.W.requires_grad = True
                else:
                    self.W_id = torch.cat([self.W_id, padded_w_id])
                    self.W = torch.cat([self.W, padded_w])
                    self.W.requires_grad = True


    def execute_on_batch(self, batch, batch_lens=None, is_sequential=None, isfold=False, foldaccumulator=None):
        #print("line 193 frist")
        #print(batch)
        # get parameters
        pass_dict = {'batch':batch}
        if batch_lens is not None:
            pass_dict['batch_lens'] = batch_lens
        if is_sequential is not None:
            pass_dict['is_sequential'] = is_sequential

        # no iterative operation
        valid_pos = [p_id.item() for p_id in torch.where(self.W_id==1)[0]]
        if not isfold:
            results = self.to_node.execute_on_batch(self.type_sign, valid_pos, **pass_dict)
            # plus weight
            if results != []: 
                global arch_search
                rand_prob = random.uniform(0, 1)
                if arch_search or rand_prob > self.randset:
                    Ws = self.softmax(self.W[self.W_id==1])
                else:
                    Ws = self.softmax(torch.ones_like(self.W)[self.W_id==1])

                for shape_id in range(len(results.shape)-1):
                    Ws = Ws.unsqueeze(-1)
                results = torch.sum(Ws * results, dim=0).contiguous()
            return results
        # iterative opereation
        else:
            assert foldaccumulator is not None
            results = self.execute_fold_function(batch, foldaccumulator)
            return results


    def execute_fold_function(self, batch, accumulator):
        batch_size, seq_len, feature_dim = batch.size()
        batch = batch.transpose(0,1) # (seq_len, batch_size, feature_dim)
        valid_pos = [p_id.item() for p_id in torch.where(self.W_id==1)[0]]

        fold_out = []
        folded_val = accumulator.clone().detach().requires_grad_(True)
        folded_val = folded_val.unsqueeze(0).repeat(batch_size,1).to(self.device)
        folded_val = folded_val.unsqueeze(0).repeat(torch.sum(self.W_id), 1, 1)
        for t in range(seq_len):
            features = batch[t].unsqueeze(0).repeat(torch.sum(self.W_id), 1, 1)
            pass_dict = {'batch':torch.cat([features, folded_val], dim=2), 'isfold':True}
            out_val = self.to_node.execute_on_batch(self.type_sign, valid_pos, **pass_dict)
            fold_out.append(out_val.unsqueeze(2))
            folded_val = out_val

        fold_out = torch.cat(fold_out, dim=2)
        # plus weight
        global arch_search
        rand_prob = random.uniform(0, 1)
        if arch_search or rand_prob > self.randset:
            Ws = self.softmax(self.W[self.W_id==1])
        else:
            Ws = self.softmax(torch.ones_like(self.W)[self.W_id==1])

        for shape_id in range(len(fold_out.shape)-1):
            Ws = Ws.unsqueeze(-1)
        fold_out = torch.sum(Ws * fold_out, dim=0).contiguous()

        return fold_out


    def get_candidates_num(self):
        return torch.sum(self.W_id)

    # clone
    def copy_self(self):
        new_edge = Edge(self.from_node, self.to_node, self.type_sign, self.device, self.randset)
        return new_edge


# node in graph (contain candidate programs)
class ProgramNode(nn.Module):
    def __init__(self, programs, depth, type_sign):
        super(ProgramNode, self).__init__()

        self.prog_dict = {type_sign:programs}
        self.new_prog = None
        self.depth = depth
        self.temp_results = {}
        self.temp_input = {}

    def extend_sign(self, new_type_sign, new_programs):
        assert new_type_sign not in self.prog_dict
        self.prog_dict[new_type_sign] = new_programs

    # reduce current branches and save new programs into temporary dict
    def temp_reduce_prog(self, topN_idxs, type_sign):
        # init
        if self.new_prog is None:
            self.new_prog = {}
        if type_sign not in self.new_prog:
            self.new_prog[type_sign] = []
        # add new
        new_idxs = []
        for idx in topN_idxs:
            if self.prog_dict[type_sign][idx] not in self.new_prog[type_sign]:
                self.new_prog[type_sign].append(self.prog_dict[type_sign][idx])
                new_idxs.append(len(self.new_prog[type_sign])-1)
            else:
                idx = self.new_prog[type_sign].index(self.prog_dict[type_sign][idx])
                new_idxs.append(idx)

        return new_idxs

    # accept temporary dict
    def do_update(self):
        self.prog_dict = self.new_prog
        self.new_prog = None

    # delete programs
    def delete_progs(self, type_sign, del_ids):
        # get new
        del_ids = np.sort(del_ids)
        new_progs = self.prog_dict[type_sign][:del_ids[0]]
        if len(del_ids) != 1:
            for cur_id, next_id in zip(del_ids[:-1], del_ids[1:]):
                new_progs = new_progs + self.prog_dict[type_sign][cur_id+1:next_id]
        new_progs = new_progs + self.prog_dict[type_sign][del_ids[-1]+1:]

        # save
        self.prog_dict[type_sign] = new_progs

    # execute
    def execute_on_batch(self, type_sign, prog_ids, **kwarg):
        #print("line 317 fuirst")
        # get keys
        #print("Kwarg is ", kwarg)
        batch_id = id(kwarg['batch'])
        is_sequential = False if 'is_sequential' not in kwarg else kwarg['is_sequential']
        is_fold = False if 'isfold' not in kwarg else kwarg['isfold']
        input_key = (batch_id, is_sequential, is_fold)

        # return stored result
        if type_sign in self.temp_results:
            if input_key in self.temp_results[type_sign]:
                # get result
                result = []
                for b_id, p_id in enumerate(prog_ids):
                    if p_id not in self.temp_results[type_sign][input_key]:
                        prog = self.prog_dict[type_sign][p_id]
                        if is_fold:
                            self.temp_results[type_sign][input_key][p_id] = prog.execute_on_batch(kwarg['batch'][b_id]).unsqueeze(0)
                        else:
                            self.temp_results[type_sign][input_key][p_id] = prog.execute_on_batch(**kwarg).unsqueeze(0)
                    result.append(self.temp_results[type_sign][input_key][p_id])
                # return
                return torch.cat(result, dim=0).contiguous()
        else:
            self.temp_results[type_sign] = {}

        # execute program
        result = []
        self.temp_results[type_sign][input_key] = {}
        if is_fold: # this is not happening
            for b_id, p_id in enumerate(prog_ids):
                prog = self.prog_dict[type_sign][p_id]
                cur_result = prog.execute_on_batch(kwarg['batch'][b_id]).unsqueeze(0)
                result.append(cur_result)
                self.temp_results[type_sign][input_key][p_id] = cur_result
        else:
            #print("not isfold")
            #print("Prog ids are " , prog_ids)
            for p_id in prog_ids:
                prog = self.prog_dict[type_sign][p_id]
                #print("Next prog is ", prog)
                cur_result = prog.execute_on_batch(**kwarg)
                if cur_result != []: # again to get around this weird, not always reproducible error
                    cur_result = prog.execute_on_batch(**kwarg).unsqueeze(0)
                #print("Current result is ", cur_result) TODO: there is some problem here I don't understand
                if cur_result != []:
                    result.append(cur_result)
                self.temp_results[type_sign][input_key][p_id] = cur_result
        #print("Result is ", result)
        # return
        # I could not resolve the error mentioned below, 2/19
        if result != []: # to get around some weird bug that's been happening where the result is empty? never gets past start function in grammar
            result = torch.cat(result, dim=0).contiguous()
        if type_sign not in self.temp_input:
            self.temp_input[type_sign] = {}
        self.temp_input[type_sign][input_key] = kwarg['batch']

        return result


    # clear result memory
    def clear_memory(self):
        self.temp_results = {}
        self.temp_input = {}

    # clone
    def copy_self(self):
        first_key = list(self.prog_dict.keys())[0]
        new_node = ProgramNode(self.prog_dict[first_key], self.depth, first_key)
        for type_sign in new_node.prog_dict:
            if type_sign != first_key:
                new_node.extend_sign(type_sign, self.prog_dict[type_sign])

        return new_node

    # get programs
    def get_programs(self):
        type_sign_list = sorted(list(self.prog_dict.keys()))
        programs = []
        for type_sign in type_sign_list:
            programs = programs + self.prog_dict[type_sign]

        return programs

    # get structure cost
    def get_program_cost(self, type_sign, prog_ids):
        #print("Type sign is " + str(type_sign))
        #print("Program ids are " + str(prog_ids))
        programs = self.prog_dict[type_sign]
        #for p_id in prog_ids:
        #    print("program is " + str(programs[p_id]) + " and the submodules are " + str(programs[p_id].submodules))
        structure_costs = [float(max(0.0, len(programs[p_id].submodules))) for p_id in prog_ids]
        #print("Structure Costs are " + str(structure_costs))
        #assert False
        return structure_costs

    # clear execution programs
    def clear_execution_progs(self):
        self.exec_prog_dict = None


# program graph for dPads
class ProgramGraph(nn.Module):

    def __init__(self, dsl_dict, input_type, output_type, input_size, output_size,
                max_num_units, min_num_units, max_depth, device, ite_beta=1.0,
                cfg=None, var_ids=None, root_symbol=None):
        super(ProgramGraph, self).__init__()

        # cost (debug)
        self.cost = 0
        self.device = device

        self.dsl_dict = dsl_dict

        self.input_type = input_type
        self.output_type = output_type
        self.input_size = input_size
        self.output_size = output_size
        self.max_num_units = max_num_units
        self.min_num_units = min_num_units
        self.max_depth = max_depth
        self.ite_beta = ite_beta

        self.cfg = cfg
        self.var_ids = var_ids

        # init root node
        self.root_node = None
        self.init_root_node(input_type, output_type, input_size, output_size, root_symbol)

    def init_root_node(self, input_type, output_type, input_size, output_size, root_symbol):
        root_fun = dsl.StartFunction(input_type=input_type, output_type=output_type,
                                     input_size=input_size, output_size=output_size,
                                     num_units=self.max_num_units, root_symbol=root_symbol)
        self.root_node = ProgramNode([root_fun], 0, type_sign=(input_type, output_type, input_size, output_size))


    # parameter for architecture
    def get_arch_params(self):
        if self.root_node is None:
            return None

        # search edges and get parameters
        arch_params = []
        param_nums = 0
        queue = [self.root_node]
        visited_param = []
        while len(queue) > 0:
            cur_node = queue.pop(0)
            # programs
            programs = cur_node.get_programs()
            # get parameters
            for prog in programs:
                for edge in prog.get_submodules().values():
                    if isinstance(edge, Edge):
                        arch_params.append(edge.get_params())
                        param_nums += arch_params[-1].view(-1).shape[0]
                        # debug
                        if id(edge.get_params()) in visited_param:
                            print('oh ho? wrong wrong')
                        visited_param.append(id(edge.get_params()))
                        # next
                        next_node = edge.get_next_node()
                        if next_node not in queue:
                            queue.append(next_node)

        return arch_params, param_nums


    # parameters for model
    def get_model_params(self):
        if self.root_node is None:
            return None

        # search edges and get parameters
        model_params = []
        param_nums = 0
        queue = [self.root_node]
        while len(queue) > 0:
            cur_node = queue.pop(0)
            # programs
            programs = cur_node.get_programs()
            # get parameters
            for prog in programs:
                # neural program
                if issubclass(type(prog), dsl.HeuristicNeuralFunction):
                    for param in prog.model.parameters():
                        model_params.append({'params': param})
                    # model_params.append({'params': prog.model.parameters()})
                    param_nums += sum([param.view(-1).shape[0] for param in list(prog.model.parameters())])
                # params of program
                elif prog.has_params:
                    for param in prog.parameters.values():
                        model_params.append({'params' : param})
                    # model_params.append({'params': list(prog.parameters.values())})
                    param_nums += sum([param.view(-1).shape[0] for param in list(prog.parameters.values())])
                # edge
                for edge in prog.get_submodules().values():
                    if isinstance(edge, Edge):
                        next_node = edge.get_next_node()
                        if next_node not in queue:
                            queue.append(next_node)
                    elif issubclass(type(edge), dsl.HeuristicNeuralFunction):
                        for param in edge.model.parameters():
                            model_params.append({'params': param})
                        # model_params.append({'params': edge.model.parameters()})
                        param_nums += sum([param.view(-1).shape[0] for param in list(edge.model.parameters())])

        return model_params, param_nums


    # reset parameter of architecture
    def reset_arch_params(self):
        with torch.no_grad():
            arch_params, _ = self.get_arch_params()
            for param in arch_params:
                param.zero_()


    def num_units_at_depth(self, depth):
        num_units = max(int(self.max_num_units*(0.5**(depth-1))), self.min_num_units)
        return num_units


    def construct_candidates(self, input_type, output_type, input_size, output_size, num_units, sem = "arith"):
        candidates = []
        replacement_candidates = self.dsl_dict[(input_type, output_type)]
        #print(replacement_candidates)
        for functionclass in replacement_candidates:
            #print("Fucntionclass is ", functionclass)
            #print("Fucntioncalss keys are", functionclass.__name__)
            if True: #Testing branch 12/10
                candidate = functionclass(input_size, output_size, num_units, sem)
                candidates.append(candidate)
            else:
                if issubclass(functionclass, dsl.ITE): # this doesn't happen I don'tthinkl
                    candidate = functionclass(input_type, output_type, input_size, output_size, num_units, beta=self.ite_beta)
                else:
                    if functionclass.__name__ == "AffineFunc":
                        candidate = functionclass(input_size, output_size, num_units, sem, ["x", "y"] ) # in theory these funcsyms shouldnt matter
                        candidates.append(candidate)
                    elif functionclass.__name__ == "LogicAndFunction": # because the rules thing is gone
                        candidate1 = functionclass(input_size, output_size, num_units, sem, [["affine", "and", "or"], ["affine", "and", "or"]])
                        #candidate2 = functionclass(input_size, output_size, num_units, sem, ["affine", "and"])
                        #candidate3 = functionclass(input_size, output_size, num_units, sem, ["affine", "or"])
                        #candidate6 = functionclass(input_size, output_size, num_units, sem, ["and", "and"])
                        #candidate7 = functionclass(input_size, output_size, num_units, sem, ["and", "or"])
                        #candidate8 = functionclass(input_size, output_size, num_units, sem, ["or", "or"])
                        candidates.append(candidate1)
                        #candidates.append(candidate2)
                        #candidates.append(candidate3)
                        #candidates.append(candidate6)
                        #candidates.append(candidate7)
                        #candidates.append(candidate8)
                    elif functionclass.__name__ == "LogicOrFunction": # redundant to above, but it is all good
                        # Note: the funcsyms I'm adding here are probably of a bad form, so I've changed the library functions a little bit to handle this
                        candidate1 = functionclass(input_size, output_size, num_units, sem, [["affine", "and", "or"], ["affine", "and", "or"]])
                        #candidate2 = functionclass(input_size, output_size, num_units, sem, ["affine", "and"])
                        #candidate3 = functionclass(input_size, output_size, num_units, sem, ["affine", "or"])
                        #candidate6 = functionclass(input_size, output_size, num_units, sem, ["and", "and"])
                        #candidate7 = functionclass(input_size, output_size, num_units, sem, ["and", "or"])
                        #candidate8 = functionclass(input_size, output_size, num_units, sem, ["or", "or"])
                        candidates.append(candidate1)
                        #candidates.append(candidate2)
                        #candidates.append(candidate3)
                        #candidates.append(candidate6)
                        #candidates.append(candidate7)
                        #candidates.append(candidate8)
        #print(candidates)
        #assert False
        return candidates

    # search the dsl to find rules to extend sub_grammar_symbol
    def construct_candidates_by_grammar_symbol(self, grammar_symbol, input_type, output_type, input_size, output_size, num_units, sem="arith"):
        candidates = []
        rules = self.cfg.productions.get(grammar_symbol)
        # print (f'grammar_symbol: {grammar_symbol}')
        for rule in rules:
            fun_name, fun_args = rule[0], rule[1:]
            if (rule[1:] == []):
                sel_id = self.var_ids[fun_name]
                candidates.append(VarSelFunction (input_size, output_size, num_units, sel_id, fun_name))
            elif fun_name in ["and", "or", "affine"]:
                # These functions have two approximation methods see library_functions.py
                functionclass = lib_funcions[fun_name]
                candidate = functionclass (input_size, output_size, num_units, sem, *fun_args)
                candidates.append(candidate)
            else:
                functionclass = lib_funcions[fun_name]
                candidate = functionclass (input_size, output_size, num_units, *fun_args)
                candidates.append(candidate)
        return candidates

    # extend node of current depth
    def extend_node(self, cur_node, depth, device, sem="arith"):
        # if reach max depth
        assert cur_node.depth <= self.max_depth
        if cur_node.depth == self.max_depth:
            return []

        # get programs
        candidate_nodes = []
        for type_sign, programs in cur_node.prog_dict.items():
            for prog_id, prog in enumerate(programs):
                # if leaf
                if len(prog.get_submodules()) == 0:
                    continue

                # find node for each submodule
                replace_submodules = {}
                sub_grammar_symbols = prog.get_sub_grammar_symbols()
                #print("Sub grammar symbols are", sub_grammar_symbols) CHANGED 12/10
                if True: #sub_grammar_symbols is None: #TODO: Changed on 12/10
                    #assert False
                    for sub_id, subm in enumerate(prog.get_submodules().items()):
                        # key and item
                        sub_key, sub_prog = subm
                        # unextended non-terminal node should connect to NN
                        if not issubclass(type(sub_prog), dsl.HeuristicNeuralFunction):
                            pdb.set_trace()
                        assert issubclass(type(sub_prog), dsl.HeuristicNeuralFunction)
                        # new program
                        sub_type_sign = (sub_prog.input_type, sub_prog.output_type, sub_prog.input_size, sub_prog.output_size)
                        require_new_progs = len(candidate_nodes) <= sub_id or sub_type_sign not in candidate_nodes[sub_id].prog_dict
                        if require_new_progs:
                            new_progs = self.construct_candidates(sub_prog.input_type, sub_prog.output_type, \
                                                                sub_prog.input_size, sub_prog.output_size, \
                                                                self.num_units_at_depth(depth))
                            # if reach max depth, program should terminate
                            if depth == self.max_depth:
                                new_leaf_progs = [prog for prog in new_progs if len(prog.get_submodules()) == 0]
                                new_progs = new_leaf_progs
                            if len(new_progs) == 0:
                                continue
                            # update node
                            if len(candidate_nodes) <= sub_id:
                                candidate_nodes.append(ProgramNode(new_progs, depth, sub_type_sign))
                            elif sub_type_sign not in candidate_nodes[sub_id].prog_dict:
                                candidate_nodes[sub_id].extend_sign(sub_type_sign, new_progs)
                        # create edge
                        new_edge = Edge(from_node=cur_node, to_node=candidate_nodes[sub_id], type_sign=sub_type_sign, device=device)
                        replace_submodules[sub_key] = new_edge
                else:
                    for sub_id, subm in enumerate(sub_grammar_symbols.items()):
                        # key and item
                        sub_key, sub_grammar_symbol = subm
                        # print (f'sub_key = {sub_key} sub_grammar_symbol = {sub_grammar_symbol}')
                        sub_prog = prog.get_submodules()[sub_key]
                        # new program
                        sub_type_sign = (sub_prog.input_type, sub_prog.output_type, sub_prog.input_size, sub_prog.output_size)
                        require_new_progs = len(candidate_nodes) <= sub_id or sub_type_sign not in candidate_nodes[sub_id].prog_dict
                        if require_new_progs:
                            new_progs = self.construct_candidates_by_grammar_symbol(sub_grammar_symbol, \
                                    sub_prog.input_type, sub_prog.output_type, \
                                    sub_prog.input_size, sub_prog.output_size, \
                                    self.num_units_at_depth(depth), sem=sem)
                            # if reach max depth, program should terminate
                            if depth == self.max_depth:
                                new_leaf_progs = [prog for prog in new_progs if len(prog.get_submodules()) == 0]
                                new_progs = new_leaf_progs
                            if len(new_progs) == 0:
                                continue
                            # update node
                            if len(candidate_nodes) <= sub_id:
                                candidate_nodes.append(ProgramNode(new_progs, depth, sub_type_sign))
                            elif sub_type_sign not in candidate_nodes[sub_id].prog_dict:
                                candidate_nodes[sub_id].extend_sign(sub_type_sign, new_progs)
                        # create edge
                        new_edge = Edge(from_node=cur_node, to_node=candidate_nodes[sub_id], type_sign=sub_type_sign, device=device)
                        replace_submodules[sub_key] = new_edge
                # new submodule
                # problem place, two edge connect to same node
                if len(replace_submodules) != 0:
                    cur_node.prog_dict[type_sign][prog_id].set_submodules(replace_submodules)

        return candidate_nodes

    # extend node of current depth (without node share)
    def extend_node_identity(self, cur_node, depth, device, sem="arith"):
        # if reach max depth
        assert cur_node.depth <= self.max_depth
        if cur_node.depth == self.max_depth:
            return []

        # get programs
        candidate_nodes = []
        for type_sign, programs in cur_node.prog_dict.items():
            for prog_id, prog in enumerate(programs):
                # if leaf
                if len(prog.get_submodules()) == 0:
                    continue

                # find node for each submodule
                replace_submodules = {}
                sub_grammar_symbols = prog.get_sub_grammar_symbols()
                if sub_grammar_symbols is None:
                    for sub_id, subm in enumerate(prog.get_submodules().items()):
                        # key and item
                        sub_key, sub_prog = subm
                        # unextended non-terminal node should connect to NN
                        assert issubclass(type(sub_prog), dsl.HeuristicNeuralFunction)
                        # new program
                        sub_type_sign = (sub_prog.input_type, sub_prog.output_type, sub_prog.input_size, sub_prog.output_size)
                        # new program
                        new_progs = self.construct_candidates(sub_prog.input_type, sub_prog.output_type, \
                                                            sub_prog.input_size, sub_prog.output_size, \
                                                            self.num_units_at_depth(depth))
                        # if reach max depth, program should terminate
                        if depth == self.max_depth:
                            new_leaf_progs = [prog for prog in new_progs if len(prog.get_submodules()) == 0]
                            new_progs = new_leaf_progs
                        if len(new_progs) == 0:
                            continue
                        # update node
                        candidate_nodes.append(ProgramNode(new_progs, depth, sub_type_sign))
                        # create edge
                        new_edge = Edge(from_node=cur_node, to_node=candidate_nodes[-1], type_sign=sub_type_sign, device=device)
                        replace_submodules[sub_key] = new_edge
                else:
                    for sub_id, subm in enumerate(sub_grammar_symbols.items()):
                        # key and item
                        sub_key, sub_grammar_symbol = subm
                        print (f'sub_key = {sub_key} sub_grammar_symbol = {sub_grammar_symbol}')
                        sub_prog = prog.get_submodules()[sub_key]
                        # new program
                        sub_type_sign = (sub_prog.input_type, sub_prog.output_type, sub_prog.input_size, sub_prog.output_size)
                        new_progs = self.construct_candidates_by_grammar_symbol(sub_grammar_symbol, \
                            sub_prog.input_type, sub_prog.output_type, \
                                sub_prog.input_size, sub_prog.output_size, \
                                self.num_units_at_depth(depth), sem=sem)
                        # if reach max depth, program should terminate
                        if depth == self.max_depth:
                            new_leaf_progs = [prog for prog in new_progs if len(prog.get_submodules()) == 0]
                            new_progs = new_leaf_progs
                        if len(new_progs) == 0:
                            continue
                        # update node
                        candidate_nodes.append(ProgramNode(new_progs, depth, sub_type_sign))
                        # create edge
                        new_edge = Edge(from_node=cur_node, to_node=candidate_nodes[-1], type_sign=sub_type_sign, device=device)
                        replace_submodules[sub_key] = new_edge

                # new submodule
                # problem place, two edge connect to same node
                if len(replace_submodules) != 0:
                    cur_node.prog_dict[type_sign][prog_id].set_submodules(replace_submodules)

        return candidate_nodes


    # function for creating cell (sub-graph) of certain depth
    def build_next_cell(self, cur_node, cell_depth, device, node_share=True, sem="arith"):
        # check depth
        assert cur_node.depth <= self.max_depth
        if cur_node.depth == self.max_depth:
            return []

        # grow graph
        cur_depth = cur_node.depth
        queue = [cur_node]
        last_nodes = []

        while len(queue) != 0:
            # curret parent node
            cur_node = queue.pop(0)
            depth = cur_node.depth
            if depth - cur_depth == cell_depth:
                break
            # extend current node
            if node_share:
                next_nodes = self.extend_node(cur_node, depth+1, device, sem=sem)
            else:
                next_nodes = self.extend_node_identity(cur_node, depth+1, device, sem=sem)

            if depth - cur_depth + 1 == cell_depth:
                last_nodes = last_nodes + next_nodes
            queue = queue + next_nodes

        return last_nodes


    # reduce and keep topN edge/nextNode
    def reduce_candidate(self, cur_node, topN, penalty=0.0):
        # programs
        finish_edges = []
        finish_nodes = []
        for cur_progs in cur_node.prog_dict.values():
            for prog in cur_progs:
                for edge in prog.get_submodules().values():
                    if isinstance(edge, Edge):
                        next_node = edge.get_next_node()
                        # update node
                        topN_idxs = edge.reduce_weights(topN, penalty)
                        new_idxs = next_node.temp_reduce_prog(topN_idxs, edge.type_sign)
                        # update weight id
                        total_num = len(next_node.new_prog[edge.type_sign])
                        edge.pad_weights(total_num, select_idxs=new_idxs)
                        # store
                        finish_edges.append(edge)
                        if next_node not in finish_nodes:
                            finish_nodes.append(next_node)

        # further update edge weight id
        for edge in finish_edges:
            next_node = edge.get_next_node()
            # update weight id
            total_num = len(next_node.new_prog[edge.type_sign])
            if edge.W.shape[0] != total_num:
                edge.pad_weights(total_num, head=False)

        # accept
        for node in finish_nodes:
            node.do_update()


    # get all nodes by BFS
    def get_all_nodes(self):
        if self.root_node is None:
            return {}
        # BFS to get all nodes
        node_dicts = {}
        queue = [self.root_node]
        while len(queue) != 0:
            cur_node = queue.pop(0)
            # store
            cur_depth = cur_node.depth
            if cur_depth not in node_dicts:
                node_dicts[cur_depth] = []
            assert cur_node not in node_dicts[cur_depth]
            node_dicts[cur_depth].append(cur_node)
            # programs
            programs = cur_node.get_programs()
            # next nodes
            for prog in programs:
                # edge
                for edge in prog.get_submodules().values():
                    if isinstance(edge, Edge):
                        next_node = edge.get_next_node()
                        if next_node not in queue:
                            queue.append(next_node)

        return node_dicts


    # get current depth
    def get_current_depth(self):
        node_dicts = self.get_all_nodes()
        depth_list = list(node_dicts.keys())
        return max(depth_list)


    # clean candidate: clean branches that unable to terminate within max depth
    # should only be called when max depth is reached
    def clean_candidate(self):
        assert self.root_node is not None
        # BFS to get all nodes
        node_dicts = self.get_all_nodes()

        # update edge (check from bottom to top)
        print (f'self.max_depth = {self.max_depth}')
        assert max(list(node_dicts.keys())) == self.max_depth
        for depth in range(self.max_depth+1)[::-1]:
            for node in node_dicts[depth]:
                # get programs
                for type_sign, prog_list in node.prog_dict.items():
                    for prog_id, prog in enumerate(prog_list):
                        sub_modules = prog.get_submodules()
                        # terminate program
                        if len(sub_modules) == 0:
                            continue
                        # check NN and whether not exists
                        invalids = [1 for subm in sub_modules.values()
                                if issubclass(type(subm), dsl.HeuristicNeuralFunction) or subm.type_sign not in subm.to_node.prog_dict]
                        if len(invalids) > 0:
                            node.prog_dict[type_sign][prog_id] = None
                            continue
                        # update edge
                        for sub_key, subm in sub_modules.items():
                            del_ids = [prog_id for prog_id, prog in enumerate(subm.to_node.prog_dict[subm.type_sign]) if prog is None]
                            if len(del_ids) != 0:
                                node.prog_dict[type_sign][prog_id].get_submodules()[sub_key].delete_weights(del_ids)
                            # check empty
                            assert len(node.prog_dict[type_sign][prog_id].get_submodules()[sub_key].W) != 0
                # check whether delete type_sign
                type_sign_list = list(node.prog_dict.keys())
                for type_sign in type_sign_list:
                    check_del = [1 for prog in node.prog_dict[type_sign] if prog is not None]
                    if len(check_del) == 0:
                        del node.prog_dict[type_sign]

        # update node
        for depth in node_dicts:
            for node in node_dicts[depth]:
                for type_sign in node.prog_dict:
                    del_ids = [prog_id for prog_id, prog in enumerate(node.prog_dict[type_sign]) if prog is None]
                    if len(del_ids) != 0:
                        node.delete_progs(type_sign, del_ids)



    # find nan programs for value
    def find_nan_progs_value(self, nan_max, **kwargs):
        # found nan nodes
        re_execute = True if len(self.root_node.temp_results) == 0 else False
        nan_nodes = self.trace_NaN_nodes(re_execute, nan_max=nan_max, kwargs=kwargs)
        # find nan programs
        nan_progs_id = {}
        depth_list = sorted([depth for depth in nan_nodes if len(nan_nodes[depth])!=0])
        for depth in depth_list[len(depth_list)-1::-1]:
            nan_progs_id[depth] = {}
            for node_id, node in enumerate(nan_nodes[depth]):
                # find Nan program by checking temp results
                cur_progs_id = {}
                for type_sign in node.temp_results:
                    program_num = len(node.prog_dict[type_sign])
                    check_nan_all = []
                    for result_dict in node.temp_results[type_sign].values():
                        check_nan = []
                        for p_id in range(program_num):
                            if p_id not in result_dict:
                                check_nan.append(torch.tensor([0]).to(self.device))
                            else:
                                check_nan.append(torch.mean(result_dict[p_id].view(1,-1).abs(), dim=1))
                        # check
                        check_nan = torch.cat(check_nan)
                        check_nan = torch.logical_or(check_nan != check_nan, check_nan >= nan_max)
                        if torch.sum(check_nan) == 0:
                            continue
                        # store
                        check_nan_all.append(check_nan.unsqueeze(0))
                    # get id
                    if len(check_nan_all) != 0:
                        check_nan_all = torch.sum(torch.cat(check_nan_all, dim=0), dim=0)
                        progs_id = torch.where(check_nan_all > 0)[0].tolist()
                        cur_progs_id[type_sign] = progs_id
                # store
                nan_progs_id[depth][node_id] = cur_progs_id

        return nan_nodes, nan_progs_id

    # early cut edges when nan exists
    def early_cut_nan(self, nan_max, **kwargs):
        if self.root_node is None:
            return
        # found nan nodes and programs
        re_execute = True if len(self.root_node.temp_results) == 0 else False
        nan_nodes, nan_progs_id = self.find_nan_progs_value(nan_max, kwargs=kwargs)
        depth_list = sorted([depth for depth in nan_nodes if len(nan_nodes[depth])!=0])

        # clean edge
        delete_upper = {}
        for depth in depth_list[::-1]:
            # nan node of current depth
            for node_id, node in enumerate(nan_nodes[depth]):
                # programs of current nan node
                cur_progs_id = nan_progs_id[depth][node_id]
                # type sign of current current program list
                for type_sign in cur_progs_id:
                    # for each program
                    for prog_id in cur_progs_id[type_sign]:
                        prog = node.prog_dict[type_sign][prog_id]
                        # get edge
                        upper = False
                        del_ids = []
                        # if leaf, directly delete upper
                        if len(prog.get_submodules()) == 0:
                            upper = True
                        # else, find delete id
                        for subm in prog.get_submodules().values():
                            # only NN for next, should happen only last depth
                            if not isinstance(subm, Edge):
                                assert depth == depth_list[-1]
                                upper = True
                                break
                            # other depth
                            try:
                                assert depth != depth_list[-1]
                            except:
                                pdb.set_trace()
                            next_node = subm.get_next_node()
                            if next_node not in nan_nodes[depth+1]:
                                continue
                            next_node_id = nan_nodes[depth+1].index(next_node)
                            # check upper delete
                            if depth+1 not in delete_upper:
                                continue
                            if next_node_id not in delete_upper[depth+1]:
                                continue
                            if subm.type_sign not in delete_upper[depth+1][next_node_id]:
                                continue
                            next_nan_ids = delete_upper[depth+1][next_node_id][subm.type_sign]
                            # clean edge
                            edge_wids = subm.W_id
                            #  if not all, clean current edge
                            if torch.sum(edge_wids[next_nan_ids]) != torch.sum(edge_wids):
                                del_ids.append(next_nan_ids)
                            # if all clean upper edge
                            else:
                                upper = True
                                break
                        # do delete
                        if not upper:
                            for del_id, subm in zip(del_ids, prog.get_submodules().values()):
                                subm.select_weights(del_id)
                        else:
                            if depth not in delete_upper:
                                delete_upper[depth] = {}
                            if node_id not in delete_upper[depth]:
                                delete_upper[depth][node_id] = {}
                            if type_sign not in delete_upper[depth][node_id]:
                                delete_upper[depth][node_id][type_sign] = []
                            delete_upper[depth][node_id][type_sign].append(prog_id)

        # clear
        if re_execute:
            self.clear_graph_results()
        # check graph save after cut
        assert self.check_graph()

    # check nodes get NaN values
    def trace_NaN_nodes(self, re_execute, nan_max=100, **kwargs):
        assert self.root_node is not None

        # execute through graph
        with torch.no_grad():
            # if temp result has been cleared, re-execute
            if re_execute:
                batch = kwargs['batch']
                output_type = kwargs['output_type']
                output_size = kwargs['output_size']
                device = kwargs['device']

                batch_input = [torch.tensor(traj) for traj in batch]
                batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
                batch_padded = batch_padded.to(device)
                # execute through graph
                type_sign = list(self.root_node.prog_dict.keys())[0]
                self.root_node.execute_on_batch(type_sign=type_sign, prog_ids=[0], batch=batch_padded, batch_lens=batch_lens)

            # check node
            nan_nodes_dict = {}
            all_nodes = self.get_all_nodes()
            for depth, node_list in all_nodes.items():
                nan_nodes_dict[depth] = []
                # check each node
                for node in node_list:
                    has_nan = False
                    for type_sign in node.temp_results:
                        program_num = len(node.prog_dict[type_sign])
                        for result_dict in node.temp_results[type_sign].values():
                            # get nan value
                            check_nan = []
                            for p_id in range(program_num):
                                if p_id not in result_dict:
                                    check_nan.append(torch.tensor([0]).to(self.device))
                                else:
                                    check_nan.append(torch.mean(result_dict[p_id].view(1,-1).abs(), dim=1))
                            # compose
                            check_nan = torch.cat(check_nan)
                            test_bool = torch.logical_or(check_nan != check_nan, check_nan >= nan_max)
                            if torch.sum(test_bool) > 0:
                                has_nan = True
                                break
                        if has_nan:
                            break
                    if has_nan:
                        nan_nodes_dict[depth].append(node)

        return nan_nodes_dict


    # clear temporary results
    def clear_graph_results(self):
        # debug
        node_dict = self.get_all_nodes()
        for depth, node_list in node_dict.items():
            for node in node_list:
                node.clear_memory()


    # check graph valid
    def check_graph(self):
        # graph exist
        if self.root_node is None:
            return False
        # root edge exists
        root_program = list(self.root_node.prog_dict.values())[0][0]
        root_edge = list(root_program.get_submodules().values())[0]
        if torch.sum(root_edge.W_id) == 0:
            return False
        # else valid
        return True

    # execute graph
    def execute_graph(self, batch, output_type, output_size, device, clear_temp=True, cur_arch_train=False):
        assert self.root_node is not None
        global arch_search
        global early_clear
        arch_search = cur_arch_train
        early_clear = clear_temp
        #print("Batch in execute_graph is ", batch)
        # pad data into tensor
        Testing = True
        # The testing part of this is me trying to get it to work for implication constraints. I have put it this way so I have a reference to a working program
        if Testing:
            batch_len = len(batch)
            batch_np = np.array(batch).reshape(2*batch_len, len(batch[0][0])) #generalized from 2 dim inputs
            batch_input = [torch.tensor(traj) for traj in batch_np]
            batch_padded, batch_lens= torch.tensor(batch_np).float().to(device), batch_input[0].size(0) #.float() as np arrays auto convert to float64 = double I think
            #print("Batch stuff ! ", batch_padded, batch_lens)
            root_program = list(self.root_node.prog_dict.values())[0][0]
            #print("root program is ", root_program)
            type_sign = list(self.root_node.prog_dict.keys())[0]
            #print("type sign is ", type_sign)
            out_padded = self.root_node.execute_on_batch(type_sign = type_sign, prog_ids=[0], batch = batch_padded, batch_lens = batch_lens)
            if out_padded != []: # again to get around the torch.cat error from line ~ 360
                out_padded = torch.reshape(out_padded, (batch_len, 2)) # Hacky solution here - flatten the trajectories so I only have to call execute on batch once, calls here are a major bottleneck.
            if clear_temp:
                self.clear_graph_results()
            return out_padded

            # OLD CODE BELOW, preserved in case there's a problem I need to address

            Time1 = time.time()
            batch_input = [ [[torch.tensor(traj) for traj in item[0] ], [torch.tensor(traj2) for traj2 in item[1]]] for item in batch]
            batch_padded = batch_input
            batch_lens = [[len(ex[0]), len(ex[1])] for ex in batch]

            # This is for batch size 1
            out_padded = [([[self.root_node.execute_on_batch(type_sign = type_sign, prog_ids = [0], batch= torch.tensor([lhs]), batch_lens = 2).squeeze()] for lhs in batch_item[0]] + [[self.root_node.execute_on_batch(type_sign = type_sign, prog_ids = [0], batch= torch.tensor([rhs]), batch_lens = 2).squeeze()] for rhs in batch_item[1]]) for batch_item in batch]
            #out_padded = [[self.root_node.execute_on_batch(type_sign = type_sign, prog_ids = [0], batch= torch.tensor([lhs]), batch_lens = 2).squeeze() for lhs in batch[0][0]]] + [[self.root_node.execute_on_batch(type_sign = type_sign, prog_ids = [0], batch= torch.tensor([rhs]), batch_lens = 2) for rhs in batch[0][1]]]
            Time2 = time.time()
            print("The inside of execute graphj took", Time2 - Time1, " seconds")
            print(out_padded)
            if clear_temp:
                self.clear_graph_results()
            return out_padded
        else:
            #assert False
            batch_input = [torch.tensor(traj) for traj in batch]
            if len(batch_input[0].size()) == 1:
                batch_padded, batch_lens = torch.tensor(batch), batch_input[0].size(0)
            else:
                batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1)) 
            batch_padded = batch_padded.to(device)
            root_program = list(self.root_node.prog_dict.values())[0][0]
            type_sign = list(self.root_node.prog_dict.keys())[0]
            out_padded = self.root_node.execute_on_batch(type_sign=type_sign, prog_ids=[0], batch=batch_padded, batch_lens=batch_lens)
            #print(batch_padded)
            #print("Batch length is " + str(batch_lens))
            #print(out_padded)

        # Now for a hacky solution
        # This is for batch size 1
    
        #print(out_padded)
        assert out_padded.shape[0] == 1
        out_padded = out_padded[0]
        #print(out_padded)
        # clear memory
        if clear_temp:
            self.clear_graph_results()
        # unpad output data
        if output_type == "list":
            out_unpadded = unpad_minibatch(out_padded, batch_lens, listtoatom=(root_program.output_type=='atom'))
        else:
            out_unpadded = out_padded
        # flatten tensor
        if len(batch_input[0].size()) == 1: # item inputs as opposed to lists
            #print(out_unpadded.squeeze())
            return out_unpadded.squeeze()
        elif output_size == 1 or output_type == "list":
            return flatten_tensor(out_unpadded).squeeze()
        else:
            if isinstance(out_unpadded, list):
                out_unpadded = torch.cat(out_unpadded, dim=0).to(device)
            return out_unpadded


    # show the graph
    def show_graph(self):
        # init
        root_node = self.root_node
        queue = [root_node]
        cur_depth = -1
        cur_node_id = 0
        # BFS
        while len(queue) != 0:
            cur_node = queue.pop(0)
            # new depth
            if cur_node.depth > cur_depth:
                cur_depth += 1
                cur_node_id = 0
                log_and_print('\n---------- depth {} ------------'.format(cur_depth))
            # edge and node
            log_and_print('-------- Node {}'.format(cur_node_id))
            # type sign
            for type_sign in cur_node.prog_dict:
                log_and_print('------ type sign contain {}    debug {}'.format(type_sign, cur_node.depth))
                # program name
                for prog_id, prog in enumerate(cur_node.prog_dict[type_sign]):
                    log_and_print('---- prog {}  :  {}'.format(prog_id, prog.name))
                    # weight and connected program
                    for subm, edge in prog.get_submodules().items():
                        log_and_print('---- subm {}'.format(subm))
                        if isinstance(edge, Edge):
                            next_node = edge.to_node
                            next_progs = next_node.prog_dict[edge.type_sign]
                            edge_probs = F.softmax(edge.W)
                            for w, prob, w_id, next_prog in zip(edge.W, edge_probs, edge.W_id, next_progs):
                                # log_and_print('-- weight {}  valid {}  : {}  | '.format(w, w_id, next_prog.name), end='')
                                log_and_print('-- weight {} | {}  valid {}  : {}  | '.format(w, prob, w_id, next_prog.name))
                            log_and_print('')
                            if next_node not in queue:
                                queue.append(next_node)
                        else:
                            log_and_print('-- NN {}'.format(edge.name))
            # next node
            cur_node_id += 1


    # get nodes of a given depth RELEVEANT
    def get_nodes(self, depth):
        if self.root_node is None:
            return None
        if depth == 0:
            return [self.root_node]

        # search edges and get parameters
        nodes = []
        queue = [self.root_node]
        while len(queue) > 0:
            cur_node = queue.pop(0)
            # finish
            if cur_node.depth > depth:
                break
            # programs
            programs = cur_node.get_programs()
            # get parameters
            for prog in programs:
                # edge
                for edge in prog.get_submodules().values():
                    if isinstance(edge, Edge):
                        next_node = edge.get_next_node()
                        if next_node.depth == depth and next_node not in nodes:
                            nodes.append(next_node)
                        elif next_node not in queue:
                            queue.append(next_node)

        return nodes


    # extract program from graph (only top1 for now)
    def extract_program(self, penalty=0.0):
        # TODO
        program_node = copy.deepcopy(self.root_node)
        self.reduce_candidate(program_node, topN=1, penalty=penalty)
        # init program
        total_prog = list(program_node.prog_dict.values())[0][0]
        # update
        queue = [total_prog]
        while len(queue) != 0:
            cur_prog = queue.pop(0)
            # edge
            subm_items = cur_prog.get_submodules().items()
            if len(subm_items) == 0:
                continue
            for sub_key, subm in subm_items:
                assert isinstance(subm, Edge)
                # get next program
                next_node = subm.get_next_node()
                next_progs = next_node.prog_dict[subm.type_sign]
                assert len(next_progs) == 1
                # update and store
                cur_prog.submodules[sub_key] = next_progs[0]
                queue.append(next_progs[0])

        return total_prog


    # print program
    def show_program(self, prog_head, is_prog=True):
        # print graph directly
        if not is_prog:
            self.show_graph()
            return None
        # print programs
        return print_program(prog_head)


    # find edge to be chosen from
    def find_prob_edge(self, node_dict, start_depth):
        max_depth = min(max(node_dict.keys()), self.max_depth)
        for depth in range(start_depth, max_depth):
            new_cost = 0
            for node_id, node in enumerate(node_dict[depth]):
                # program list
                for type_sign in node.prog_dict:
                    cur_progs = node.prog_dict[type_sign]
                    for prog_id, prog in enumerate(cur_progs):
                        # edge
                        for sub_key in prog.submodules:
                            edge = prog.submodules[sub_key]
                            edge_num = edge.get_candidates_num()
                            new_cost += 1
                            if isinstance(edge, Edge) and edge_num>1:
                                return [depth, node_id, type_sign, prog_id, sub_key, edge_num, new_cost]

        return None


    # backward probability to parameters of subgraph under specific edge
    def pick_back_prob(self, prob, cur_edge):
        queue = [cur_edge.get_next_node()]
        visited_nodes = []
        while len(queue) != 0:
            cur_node = queue.pop(0)
            # debug
            assert cur_node not in visited_nodes
            visited_nodes.append(cur_node)
            # programs
            programs = cur_node.get_programs()
            for prog in programs:
                # leaf
                if prog.has_params:
                    for param_key in prog.parameters:
                        prog.parameters[param_key].data = prog.parameters[param_key].data * prob
                # non leaf
                else:
                    for edge in prog.get_submodules().values():
                        # go to next
                        if isinstance(edge, Edge):
                            next_node = edge.get_next_node()
                            if next_node not in queue:
                                queue.append(next_node)
                        # if neural
                        else:
                            assert issubclass(type(edge), dsl.HeuristicNeuralFunction)
                            for param in list(edge.model.parameters()):
                                param.data = param.data * prob

    # calculate cost based on uncertain edges (for earlier evaluation)
    def predict_cost(self, graph, depth):
        uncertain_cost = 0
        # from last
        if depth != 0:
            depth -= 1
        frontiers = graph.get_nodes(depth)
        # check candidate edges
        while len(frontiers) != 0:
            node = frontiers.pop(0)
            # program list
            program_list = node.get_programs()
            for prog in program_list:
                # edge
                for edge in prog.submodules.values():
                    # if neural model
                    if issubclass(type(edge), dsl.HeuristicNeuralFunction):
                        uncertain_cost += 1
                        continue
                    # candidate
                    candidate_num = edge.get_candidates_num()
                    if candidate_num > 1:
                        uncertain_cost += 1
                    else:
                        next_node = edge.get_next_node()
                        if next_node not in frontiers:
                            frontiers.append(next_node)

        return uncertain_cost


    # change one approximate edge into exact path
    def get_children(self, graph, start_depth, clean=False, back_prob=True):
        # init
        node_dict = graph.get_all_nodes()
        # find the edge to be choose from
        edge_pos = graph.find_prob_edge(node_dict, start_depth)
        depth, node_id, type_sign, prog_id, sub_key, edge_num, after_cost = edge_pos
        for d in range(depth+1):
            after_cost += len(node_dict[d])
        # create child for each possible edge
        final_cost = []
        children = [copy.deepcopy(graph) for i in range(edge_num)]
        for child_id, child in enumerate(children):
            # get node
            finish_edges = []
            finish_nodes = []
            cur_node = child.get_all_nodes()[depth][node_id]
            child_edge = cur_node.prog_dict[type_sign][prog_id].submodules[sub_key]
            # do build
            for cur_progs in cur_node.prog_dict.values():
                for prog in cur_progs:
                    for edge in prog.get_submodules().values():
                        if not isinstance(edge, Edge):
                            continue
                        # select for certain edge
                        if edge == child_edge:
                            topN_idxs, topN_prob = edge.select_topN_weights(child_id, sorted=False, get_prob=True)
                        else:
                            # directly add all
                            topN_idxs = edge.select_valid_weights()
                        # update next node
                        next_node = edge.get_next_node()
                        new_idxs = next_node.temp_reduce_prog(topN_idxs, edge.type_sign)
                        # update weight
                        total_num = len(next_node.new_prog[edge.type_sign])
                        try:
                            edge.pad_weights(total_num, select_idxs=new_idxs)
                        except:
                            pdb.set_trace()
                        # store
                        finish_edges.append(edge)
                        if next_node not in finish_nodes:
                            finish_nodes.append(next_node)

            # further update edge weight id
            for edge in finish_edges:
                next_node = edge.get_next_node()
                # update weight id
                total_num = len(next_node.new_prog[edge.type_sign])
                if edge.W.shape[0] != total_num:
                    edge.pad_weights(total_num, head=False)

            # accept
            for node in finish_nodes:
                node.do_update()

            # clean
            if clean:
                child.clean_candidate()

            # back probability to parameters
            if back_prob:
                with torch.no_grad():
                    self.pick_back_prob(topN_prob, child_edge)

            # add uncertain cost
            final_cost.append(after_cost + self.predict_cost(child, depth))

        return children, depth, final_cost


    # check if graph terminate
    def check_terminate(self, graph, start_depth):
        # return bool
        node_dict = graph.get_all_nodes()
        terminate = graph.find_prob_edge(node_dict, start_depth) is None
        return terminate

    # check if graph contain neural module
    def check_neural(self, graph):
        node_dict = graph.get_all_nodes()
        max_depth = max(list(node_dict.keys()))
        nodes = node_dict[max_depth]
        # check each node
        for cur_node in nodes:
            programs = []
            for prog_list in cur_node.prog_dict.values():
                programs += prog_list
            non_complete = [1 for prog in programs if len(prog.get_submodules())!=0]
            if len(non_complete) != 0:
                return True
        # no neural
        return False

    # get cost (only after terminate)
    def get_cost(self, graph):
        assert self.check_terminate(graph, 0)
        node_dict = graph.get_all_nodes()
        cost = sum([len(node_dict[depth]) for depth in node_dict])
        # uncertain cost due to neural modules
        max_depth = max(list(node_dict.keys()))
        uncertain_cost = self.predict_cost(graph, max_depth)
        return cost + uncertain_cost

    # calculate cost
    def cal_cost(self, graph):
        cost = 0
        queue = [graph.root_node]
        while len(queue) > 0:
            # new node
            cur_node = queue.pop(0)
            programs = cur_node.get_programs()
            assert len(programs) == 1
            cost += 1
            # check next
            for prog in programs:
                for edge in prog.submodules.values():
                    # if neural model
                    if issubclass(type(edge), dsl.HeuristicNeuralFunction):
                        cost += 1
                        continue
                    # candidate
                    candidate_num = edge.get_candidates_num()
                    if candidate_num > 1:
                        cost += 1
                    else:
                        next_node = edge.get_next_node()
                        if next_node not in queue:
                            queue.append(next_node)

        return cost

    # split
    def split_urgent(self):
        cur_nodes = [self.root_node]

        best_cut = None
        best_group1 = None
        best_group2 = None
        target_edge = None
        target_node = None
        while (len(cur_nodes)):
            next_nodes = []
            for node in cur_nodes:
                for key, progs in node.prog_dict.items():
                    for prog in progs:
                        for _, edge in prog.submodules.items():
                            next_nodes.append(edge.to_node)
                            weights = nn.Softmax()(edge.W)
                            n = len(weights)
                            if n > 1:
                                gradients = edge.W.grad
                                #print(gradients) TODO: changed this

                                # calculate the cosine similarity
                                num_ops = len(gradients)
                                dist_mat = torch.zeros((num_ops, num_ops))
                                for opid_i in range(num_ops):
                                    for opid_j in range(num_ops):
                                        dist_mat[opid_i, opid_j] = F.cosine_similarity(gradients[opid_i], gradients[opid_j], dim=-1)


                                # solve the mincut problem
                                first = True
                                # one group with one operation
                                for i in range(0, num_ops):
                                    list1 = [i]
                                    list2 = list(range(0, num_ops))
                                    list2.remove(i)
                                    if first:
                                        first = False
                                        group1 = list1
                                        group2 = list2
                                        cut = sum([dist_mat[list1[0]][x] for x in list2])
                                    else:
                                        new_cut = sum([dist_mat[list1[0]][x] for x in list2])
                                        if new_cut < cut:
                                            cut = new_cut
                                            group1 = list1
                                            group2 = list2

                                if best_cut is None:
                                    parent_node = node
                                    child_node = edge.to_node
                                    best_cut = cut
                                    best_group1 = group1
                                    best_group2 = group2
                                elif cut < best_cut:
                                    parent_node = node
                                    child_node = edge.to_node
                                    best_cut = cut
                                    best_group1 = group1
                                    best_group2 = group2

                                # each group has two operation
                                for i in range(0, num_ops):
                                    for j in range(0, num_ops):
                                        if i != j:
                                            list1 = [i, j]
                                            list2 = list(range(0, num_ops))
                                            list2.remove(i)
                                            list2.remove(j)

                                            new_cut = 0
                                            new_cut += sum([dist_mat[list1[0]][x] for x in list2])
                                            new_cut += sum([dist_mat[list1[1]][x] for x in list2])
                                            if new_cut < cut:
                                                cut = new_cut
                                                group1 = list1
                                                group2 = list2

            cur_nodes = next_nodes
            # now manupulate the graph
        return self.split_urgent_specific(parent_node, child_node)

    def split_urgent_specific(self, parent_node, child_node):
        for target_type_sign, target_progs in child_node.prog_dict.items():
            break
        n = len(target_progs)
        graph_copies = [self] + [copy.deepcopy(self) for _ in range(1, n)]

        cur_nodes_list = [[g.root_node] for g in graph_copies]
        place_found = False
        while(len(cur_nodes_list[0])):
            next_nodes_list = []
            for g_idx, cur_nodes in enumerate(cur_nodes_list):
                next_nodes = []
                if g_idx == 0:
                    for node_idx, node in enumerate(cur_nodes):
                        if node is parent_node:
                            for progs_idx, (type_sign, progs) in enumerate(node.prog_dict.items()):
                                for prog_idx, prog in enumerate(progs):
                                    for edge_idx, (submodule, edge) in enumerate(prog.submodules.items()):
                                        next_nodes.append(edge.to_node)
                                        if edge.to_node is child_node:
                                            position = (node_idx, progs_idx, type_sign, prog_idx, edge_idx, submodule)
                                            place_found = True
                                            edge.W_id = torch.zeros_like(edge.W_id)
                                            edge.W_id[g_idx] = 1
                else:
                    if place_found:
                        node_idx, progs_idx, type_sign, prog_idx, edge_idx, submodule = position
                        node = cur_nodes[node_idx]
                        progs = node.prog_dict[type_sign]
                        for prog_idx, prog in enumerate(progs):
                            edge = prog.submodules[submodule]
                            edge.W_id = torch.zeros_like(edge.W_id)
                            edge.W_id[g_idx] = 1
                        continue
                    for node_idx, node in enumerate(cur_nodes):
                        for progs_idx, (type_sign, progs) in enumerate(node.prog_dict.items()):
                            for prog_idx, prog in enumerate(progs):
                                for edge_idx, (submodule, edge) in enumerate(prog.submodules.items()):
                                    next_nodes.append(edge.to_node)

                next_nodes_list.append(next_nodes)
            if place_found:
                break
            cur_nodes_list = next_nodes_list
        return graph_copies

    def __lt__(self, other):
        return self.max_depth <= other.max_depth

    def partition(self, top_left, GM):
        if top_left:
            return self.split(GM)
        else:
            return self.split_urgent()

    def split(self, GM):
        # first, find the node we will split
        cur_nodes = [self.root_node]
        target_edge = None
        target_node = None
        while (len(cur_nodes)):
            next_nodes = []
            for node in cur_nodes:
                for key, progs in node.prog_dict.items():
                    #print(key)
                    for prog in progs:
                        #print(prog)
                        for _, edge in prog.submodules.items():
                            next_nodes.append(edge.to_node)
                            child_progs = list(edge.to_node.prog_dict.values())[0]
                            #print(child_progs)
                            if len(child_progs) > 1:
                                target_edge = edge
                                target_node = edge.to_node
                                if GM:
                                    return self.split_specific_with_GM(target_edge, target_node)
                                return self.split_specific(target_edge, target_node)
            #print(next_nodes)
            cur_nodes = next_nodes

    def can_split(self):
        # first, find the node we will split
        cur_nodes = [self.root_node]

        target_edge = None
        target_node = None
        while (len(cur_nodes)):
            next_nodes = []
            for node in cur_nodes:
                for key, progs in node.prog_dict.items():
                    for prog in progs:
                        for _, edge in prog.submodules.items():
                            next_nodes.append(edge.to_node)
                            child_progs = list(edge.to_node.prog_dict.values())[0]
                            if len(child_progs) > 1:
                                return True

            cur_nodes = next_nodes
        return False

    def split_specific_with_GM(self, target_edge, target_node):
        graph_copies = [self, copy.deepcopy(self)]
        gradients = target_edge.W.grad
        #print(gradients) TODO: changed this

        # calculate the cosine similarity
        num_ops = len(gradients)
        dist_mat = torch.zeros((num_ops, num_ops))
        for opid_i in range(num_ops):
            for opid_j in range(num_ops):
                dist_mat[opid_i, opid_j] = F.cosine_similarity(gradients[opid_i], gradients[opid_j], dim=-1)


        # solve the mincut problem
        first = True
        # one group with one operation
        for i in range(0, num_ops):
            list1 = [i]
            list2 = list(range(0, num_ops))
            list2.remove(i)
            if first:
                first = False
                group1 = list1
                group2 = list2
                cut = sum([dist_mat[list1[0]][x] for x in list2])
            else:
                new_cut = sum([dist_mat[list1[0]][x] for x in list2])
                if new_cut < cut:
                    cut = new_cut
                    group1 = list1
                    group2 = list2

        # each group has two operation
        for i in range(0, num_ops):
            for j in range(0, num_ops):
                if i != j:
                    list1 = [i, j]
                    list2 = list(range(0, num_ops))
                    list2.remove(i)
                    list2.remove(j)

                    new_cut = 0
                    new_cut += sum([dist_mat[list1[0]][x] for x in list2])
                    new_cut += sum([dist_mat[list1[1]][x] for x in list2])
                    if new_cut < cut:
                        cut = new_cut
                        group1 = list1
                        group2 = list2

        # now manupulate the graph

        for target_type_sign, target_progs in target_node.prog_dict.items():
            break
        # n = len(target_progs)
        # graph_copies = [self] + [copy.deepcopy(self) for _ in range(1, n)]

        cur_nodes_list = [[g.root_node] for g in graph_copies]
        place_found = False
        while(len(cur_nodes_list[0])):
            next_nodes_list = []
            for g_idx, cur_nodes in enumerate(cur_nodes_list):
                next_nodes = []
                if g_idx == 0:
                    for node_idx, node in enumerate(cur_nodes):
                        for progs_idx, (type_sign, progs) in enumerate(node.prog_dict.items()):
                            for prog_idx, prog in enumerate(progs):
                                for edge_idx, (submodule, edge) in enumerate(prog.submodules.items()):
                                    next_nodes.append(edge.to_node)
                                    if edge is target_edge:
                                        # find the place to split
                                        place_found = True
                                        position = (node_idx, progs_idx, type_sign, prog_idx, edge_idx, submodule)
                                        assert edge.to_node is target_node
                                        new_node = ProgramNode([target_progs[x] for x in group1], target_node.depth, target_type_sign)
                                        new_edge = Edge(node, new_node, edge.type_sign, edge.device, edge.randset)

                                        # copy weights
                                        lst = []
                                        for j in group1:
                                            lst.append(target_edge.W[j])
                                        new_edge.W = torch.tensor(lst, requires_grad=True, device=new_edge.device)
                                        assert len(node.prog_dict[type_sign]) == 1
                                        node.prog_dict[type_sign][0].submodules[submodule] = new_edge
                else:
                    if place_found:
                        node_idx, progs_idx, type_sign, prog_idx, edge_idx, submodule = position
                        node = cur_nodes[node_idx]
                        edge = cur_nodes[node_idx].prog_dict[type_sign][prog_idx].submodules[submodule]
                        new_node = ProgramNode([target_progs[x] for x in group2], target_node.depth, target_type_sign)
                        new_edge = Edge(node, new_node, edge.type_sign, edge.device, edge.randset)
                        # copy weight
                        lst = []
                        for j in group2:
                            lst.append(target_edge.W[j])
                        new_edge.W = torch.tensor(lst, requires_grad=True, device=new_edge.device)
                        node.prog_dict[type_sign][0].submodules[submodule] = new_edge
                        continue
                    for node_idx, node in enumerate(cur_nodes):
                        for progs_idx, (type_sign, progs) in enumerate(node.prog_dict.items()):
                            for prog_idx, prog in enumerate(progs):
                                for edge_idx, (submodule, edge) in enumerate(prog.submodules.items()):
                                    next_nodes.append(edge.to_node)

                next_nodes_list.append(next_nodes)
            if place_found:
                break
            cur_nodes_list = next_nodes_list
        return graph_copies

    def split_specific(self, target_edge, target_node):
        for target_type_sign, target_progs in target_node.prog_dict.items():
            break
        n = len(target_progs)
        graph_copies = [self] + [copy.deepcopy(self) for _ in range(1, n)]

        cur_nodes_list = [[g.root_node] for g in graph_copies]
        place_found = False
        while(len(cur_nodes_list[0])):
            next_nodes_list = []
            for g_idx, cur_nodes in enumerate(cur_nodes_list):
                next_nodes = []
                if g_idx == 0:
                    for node_idx, node in enumerate(cur_nodes):
                        for progs_idx, (type_sign, progs) in enumerate(node.prog_dict.items()):
                            for prog_idx, prog in enumerate(progs):
                                for edge_idx, (submodule, edge) in enumerate(prog.submodules.items()):
                                    next_nodes.append(edge.to_node)
                                    if edge is target_edge:
                                        # find the place to split
                                        place_found = True
                                        position = (node_idx, progs_idx, type_sign, prog_idx, edge_idx, submodule)
                                        assert edge.to_node is target_node
                                        new_node = ProgramNode([target_progs[g_idx]], target_node.depth, target_type_sign)
                                        new_edge = Edge(node, new_node, edge.type_sign, edge.device, edge.randset)
                                        assert len(node.prog_dict[type_sign]) == 1
                                        node.prog_dict[type_sign][0].submodules[submodule] = new_edge
                else:
                    if place_found:
                        node_idx, progs_idx, type_sign, prog_idx, edge_idx, submodule = position
                        node = cur_nodes[node_idx]
                        edge = cur_nodes[node_idx].prog_dict[type_sign][prog_idx].submodules[submodule]
                        new_node = ProgramNode([target_progs[g_idx]], target_node.depth, target_type_sign)
                        new_edge = Edge(node, new_node, edge.type_sign, edge.device, edge.randset)
                        node.prog_dict[type_sign][0].submodules[submodule] = new_edge
                        continue
                    for node_idx, node in enumerate(cur_nodes):
                        for progs_idx, (type_sign, progs) in enumerate(node.prog_dict.items()):
                            for prog_idx, prog in enumerate(progs):
                                for edge_idx, (submodule, edge) in enumerate(prog.submodules.items()):
                                    next_nodes.append(edge.to_node)

                next_nodes_list.append(next_nodes)
            if place_found:
                break
            cur_nodes_list = next_nodes_list
        return graph_copies
