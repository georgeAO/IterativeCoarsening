# Iterative Coarsening

A repository which implements the learning of kDGs and MkDGs 
from real and synthetic data. The approach used to learn 
the graphical structures is based on the work 
described in https://opt-ml.org/oldopt/papers/OPT2015_paper_36.pdf.

---

## Features

This work learns decomposable models through a process named *Iterative Coarsening* (IC). Additionally, we implement a greedy hill-climbing approach which learns decomposable models by adding a single edge to the model in each iteration which maximizes a local scoring function. Currently, both methods support:

- Learning with real data sets using the BDeu scoring function.
- Bouding the maximal clique size of a learned model. 
- Learning graphs which are a maximal kDG (IC only).

At present, both methods begin the learning process by using a maximal spanning tree. Hence, if the optimal graph is one which is which is very disconnected, these implementations may not obtain ideal results. However, we envision to update both methods soon to account for this. 

For obtaining an optimal result on a data set, consider using [GOBNILP](https://bitbucket.org/jamescussens/pygobnilp/src/master/). Be advised that since this is an exact approach, the computation time can be significantly larger than our approaches. 

---

## Dependencies

- NumPy
- SciPy
- GurobiPy (must be used with Gurobi v9.0 or greater).
- NetworkX
- pgmpy
- pandas

All packages available through pypi (except for Gurobi). For academic use, you may obtain a licensed version of Gurobi [here](https://www.gurobi.com/academia/academic-program-and-licenses/). 


---

## Usage

### Declare new `DecomposableModel` object:

```python
model = DecomposableModel(fileName=path_to_data, alpha=1)
```
Parameters:
* fileName : string,
    path to .csv file containing the raw data set.
* alpha : float, the equivalent sample size of the Dirichlet uniform prior

### Learn with IC `learn` method:
```python
model.learn(k_max=np.inf, l_max=np.inf, max_time=1000.0, max_time_gurobi=500.0, maximal=False)
```
Parameters:
* k_max : int (or np.inf),
    the maximal clique size of the model to learn.
* l_max : int (or np.inf), the maximal length of an edge to consider adding. Length is defined by the number of separators               between to vertices.
* max_time : float (or np.inf), the maximal running time for the entire learning algorithm.
* max_time_gurobi : float (or np.inf), the maximal running time for each call to gurobi.
* maximal : bool, Whether the graph should be a maximal kDG (MkDG) or kDG.

### Learn with hill-climbing `greedy_learn` method:
```python
model.greedy_learn(k_max=np.inf, time_limit=np.inf)
```

Parameters:
* k_max : int (or np.inf), the maximal clique size of the model to learn.
* time_limit : float (or np.inf), the maximal running time for hill-climbing. 



---

## Example 

```python
#learn a model with water1000 data using Iterative Coarsening
test = DecomposableModel("data/water1000.csv")
test.learn(k_max=4)
test.to_bn()
kdg_model = test.get_model_directed()

#relearn the model using hill-climbing
test.greedy_learn(k_max=4)
greedy_model = test.get_model_directed()

#compare results
bdeu = test.get_score_function()
print("kDG score is " + str(bdeu.score(kdg_model)))
print("greedy score is " + str(bdeu.score(greedy_model)))
```

---



- **[MIT license](http://opensource.org/licenses/mit-license.php)**
