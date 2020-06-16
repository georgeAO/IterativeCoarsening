from decomposable_model import DecomposableModel

#build the decomposable dodel object with relevant data
test = DecomposableModel("Wine.csv")
#learn the model using the kdg approach
test.learn(k_max=4)
#convert learned (undirected) model to directed with minimal I-map
test.to_bn()
kdg_model = test.get_model_directed()

#relearn the model using the greedy approach
test.greedy_learn(k_max=4)
greedy_model = test.get_model_directed()

#get the BDeu score object
bdeu = test.get_score_function()

#print results
print(bdeu.score(kdg_model))
print(bdeu.score(greedy_model))
