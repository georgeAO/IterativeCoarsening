from decomposable_model import DecomposableModel
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import chordal as ch


#build the decomposable model object with relevant data
test = DecomposableModel("data/autos.csv")
#learn the model using the kdg approach
test.learn(k_max=4)
#test.learn(k_max=np.inf, l_max=1, max_time=1000.0, max_time_gurobi=500.0, maximal=False, coarsen=True, extra_iters=2)
test.coarsen()
#convert learned (undirected) model to directed with minimal I-map
test.to_bn()
kdg_model = test.get_model_directed()
test.coarsen()

#relearn the model using the greedy approach
test.greedy_learn(k_max=4)
greedy_model = test.get_model_directed()

#get the BDeu score object
bdeu = test.get_score_function()

#print results
print("kDG score is " + str(bdeu.score(kdg_model)))
print("greedy score is " + str(bdeu.score(greedy_model)))
