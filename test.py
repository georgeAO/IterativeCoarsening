from decomposable_model import DecomposableModel

#build the decomposable dodel object with relevant data
test = DecomposableModel("water1000.csv")
#build the model using the kdg approach
test.learn()
#convert learned (undirected) model to directed with minimal I-map
test.to_bn()

kdg_model = test.get_model_directed()
bdeu = test.get_score_function()

print(bdeu.score(kdg_model))