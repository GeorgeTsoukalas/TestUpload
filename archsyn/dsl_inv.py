import dsl.library_functions as dsl

DSL_DICT = {('list', 'list') : [],
                        ('list', 'atom') : [],
                        #('atom', 'atom') : [dsl.AffineFunc, dsl.LogicAndFunction]}
                        #('atom', 'atom') : [dsl.EqualityFunc, dsl.LogicAndFunction]}
                        #('atom', 'atom') : [dsl.AffineFunc, dsl.LogicAndFunction, dsl.LogicOrFunction]}
                        ('atom', 'atom') : [dsl.AffineFunc, dsl.LogicAndFunction, dsl.LogicOrFunction, dsl.EqualityFunc]} # this one is for the equality function I added
                        #('atom', 'atom') : [dsl.AffineFunc, dsl.LogicAndFunction,  dsl.EqualityFunc]}
                        #('atom', 'atom') : [dsl.AffineFunc, dsl.EqualityFunc, dsl.LogicOrFunction]}

CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}

