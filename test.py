from decomposable_model import DecomposableModel

#build the decomposable dodel object with relevant data
test = DecomposableModel("Wine.csv")
#build the model using the kdg approach
test.learn(k_max=4)
hold = test.get_model_undirected()
#convert learned (undirected) model to directed with minimal I-map
test.to_bn()
kdg_model = test.get_model_directed()

#build the model using the greedy approach
test.greedy_learn(k_max=4)
greedy_model = test.get_model_directed()

bdeu = test.get_score_function()

print(bdeu.score(kdg_model))