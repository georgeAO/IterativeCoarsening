from pgmpy.estimators import BDeuScore
import networkx as nx
import numpy as np
import scipy as scp
import pandas as pd
import chordal as ch
import constraints as co
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations


class DecomposableModel:
    def __init__(self, fileName, alpha=1):

        self.alpha = alpha
        self.data = np.genfromtxt(fileName, delimiter=',')
        self.num_vars = self.data.shape[1]
        self.data_frame = pd.DataFrame(self.data, columns=list(range(self.num_vars)))
        self.var_states = []


        for index in range(len(self.data_frame.columns)):
            self.var_states.append(self.data_frame[index].nunique())

        self.bdeu = BDeuScore(self.data_frame, equivalent_sample_size=alpha)

        self.undirected = nx.Graph()
        self.undirected.add_nodes_from(range(self.num_vars))

        self.directed = nx.Graph()

    def mst(self):
        complete_graph = nx.Graph()
        lexicographic_edges = list(combinations(list(range(self.num_vars)), 2))
        weight_list = self.get_weight_list(lexicographic_edges)

        for index, edge in enumerate(lexicographic_edges):
            complete_graph.add_edge(edge[0], edge[1], weight=-1 * weight_list[index])

        edges = list(nx.algorithms.tree.minimum_spanning_edges(complete_graph, algorithm='kruskal', data=False))
        self.undirected.add_edges_from(edges)

    def get_score(self, V):
        '''
        The score function.

        In the case of alpha=0, it computes the entropy of a given set of variables.

        In the case of alpha>0, it computes the BDeu score with equivalent sample size equal to alpha associated to
        a set of variables: BDeu(V)= sum_v ln(gamma(N_v + alpha/r_V))- ln(gamma(alpha/r_V) where N_v denotes the number
        of cases of the dataset in which variable V takes the value v, and r_V denotes the cardinality (the number of
        different values) of the set of variables V.

        Remarks: this method only works for discrete random variables

        :param V: A subset of variables given by their indices, list(int)
        :param D: A data set, np.array(int) of dimension num_samples X num_variables.
        :param r: The number of values that can take each variable in V, i.e. r[i] is the size of the support of V[i],
        np.array(int) of dimension len(V)
        :param alpha: the equivalent sample size of the Dirichlet uniform prior
        :return: The empirical entropy of variables V given the data set D
        '''

        # Obtain the cardinality of V
        if len(V) == 0:
            return scp.special.loggamma(self.alpha) - scp.special.loggamma(self.num_vars + self.alpha)

        r = [self.var_states[i] for i in V]
        rV = np.prod(r)

        # Get the frequencies of every observed configuration
        N = np.unique(self.data[:, V], return_counts=True, axis=0)[1]
        # Get the empirical probabilities o the observed configurations
        if self.alpha > 0:
            lG = scp.special.loggamma(self.alpha / rV) - scp.special.loggamma(N + self.alpha / rV)
            # The probability of each of the (rV-len(N)) unobserved configurations of V
            # Compute the entropy (in 2 basis)
            return np.sum(lG)
        else:
            N /= self.data.shape[0]
            return -np.sum(N * np.log(N))

    def get_weight_list(self, edges):
        weightList = []
        for edge in edges:
            weightList.append(self.get_score([edge[0]]) + self.get_score([edge[1]]) - self.get_score(list(edge))
                              - self.get_score([]))
        return weightList

    def get_objective(self, dict_of_vars):
        c = np.zeros(len(dict_of_vars.keys()))
        for key in dict_of_vars.keys():
            to_scoreUVS = list({key[0][0], key[0][1]}.union(key[1]))
            to_scoreUS = list({key[0][0]}.union(key[1]))
            to_scoreVS = list({key[0][1]}.union(key[1]))
            to_scoreS = list(key[1])

            c[dict_of_vars[key]] = self.get_score(to_scoreUS) + self.get_score(to_scoreVS) - \
                                   self.get_score(to_scoreUVS) - self.get_score(to_scoreS)
        return c

    def learn(self, k_max=np.inf, l_max=np.inf, max_time=500.0, maximal=False):
        self.mst()
        current_model = self.undirected
        sol = [1]
        i = 2
        while np.linalg.norm(sol, ord=1) > 0.01 and i <= k_max:
            edge_list = ch.getSortedEdges(current_model)
            cliques = ch.chainOfCliques(current_model)
            sep_list = ch.getSeparators(cliques)
            comps = ch.getCompList(current_model, sep_list)
            sep_to_comps = dict(zip(sep_list, comps))
            e_to_seps = ch.getSepsForCandEdges(comps, sep_list)
            list_vars = co.generateDecVariables(e_to_seps, l_max)
            dict_of_vars = {uvS: ind for ind, uvS in enumerate(list_vars)}
            c = self.get_objective(dict_of_vars)

            m = gp.Model("iterate" + str(i))
            x = m.addMVar(shape=len(c), vtype=GRB.BINARY, name="x")
            m.setObjective(c @ x, GRB.MAXIMIZE)
            m.Params.MIPFocus = 1
            m.Params.timeLimit = max_time

            A1, b1 = co.type1(e_to_seps, dict_of_vars)
            m.addConstr(A1 @ x <= b1, name="c1")

            A2u, b2u, A2v, b2v = co.type2(e_to_seps, dict_of_vars, edge_list, i)
            m.addConstr(A2u @ x <= b2u, name="c2u")
            m.addConstr(A2v @ x <= b2v, name="c2v")

            A3, b3, r = co.type3(sep_to_comps, dict_of_vars)

            if maximal:
                m.addConstr(A3[0:r, :] @ x == b3[0:r], name="c3eq")
                m.addConstr(A3[r:len(b3), :] @ x <= b3[r:len(b3)], name="c3ineq")
            else:
                m.addConstr(A3 @ x <= b3, name="c3")

            m.optimize()
            sol = x.X

            new_edges = ch.updateGraph(sol, dict_of_vars)
            e = edge_list + new_edges
            current_model = nx.Graph(e)

        self.undirected = current_model
