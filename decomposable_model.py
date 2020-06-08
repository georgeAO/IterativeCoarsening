from pgmpy.estimators import BDeuScore
from pgmpy.models import MarkovModel, BayesianModel
import networkx as nx
import numpy as np
import scipy as scp
import pandas as pd
import time as time
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

    def learn(self, k_max=np.inf, l_max=np.inf, max_time=1000.0, max_time_gurobi=500.0, maximal=False):
        self.mst()
        current_model = self.undirected
        sol = [1]
        i = 2
        start = time.time()
        end = 0
        while np.linalg.norm(sol, ord=1) > 0.01 and i <= k_max and (end - start) < max_time:
            edge_list = ch.get_sorted_edges(current_model)
            cliques = ch.chain_of_cliques(current_model)
            sep_list = ch.get_separators(cliques)
            comps = ch.get_comp_list(current_model, sep_list)
            sep_to_comps = dict(zip(sep_list, comps))
            e_to_seps = ch.get_seps_for_cand_edges(comps, sep_list)
            list_vars = co.generate_dec_variables(e_to_seps, l_max)
            dict_of_vars = {uvS: ind for ind, uvS in enumerate(list_vars)}
            c = self.get_objective(dict_of_vars)

            m = gp.Model("iterate" + str(i))
            x = m.addMVar(shape=len(c), vtype=GRB.BINARY, name="x")
            m.setObjective(c @ x, GRB.MAXIMIZE)
            m.Params.MIPFocus = 1
            m.Params.timeLimit = max_time_gurobi

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

            new_edges = ch.update_graph(sol, dict_of_vars)
            e = edge_list + new_edges
            current_model = nx.Graph(e)
            end = time.time()

        self.undirected = current_model

    def to_bn(self, graph):
        # Generate connected components
        connected = list(nx.algorithms.components.connected_components(graph))
        # Hold the edges of each connected component as a list(list())
        connected_comps = list()
        # The final bn model
        final_bn = BayesianModel()
        # If the graph is not completely connected
        if len(connected) > 1:
            for comp in connected:
                # If the connected component is not a single vertex
                if len(comp) > 1:
                    edges_to_add = list()
                    temp = MarkovModel()
                    for vert1 in comp:
                        neighbours = list(graph.neighbors(vert1))
                        for vert2 in neighbours:
                            if not ((vert1, vert2) in edges_to_add) and not ((vert2, vert1) in edges_to_add):
                                edges_to_add.append((vert1, vert2))

                    temp.add_nodes_from(comp)
                    temp.add_edges_from(edges_to_add)
                    temp_bn = temp.to_bayesian_model()
                    connected_comps.append(temp_bn)
                else:
                    final_bn.add_nodes_from(comp)

            for bn in connected_comps:
                final_bn.add_nodes_from(list(bn.nodes))
                final_bn.add_edges_from(list(bn.edges))

            return final_bn

        else:
            # If the graph is completely connected, just add all edges to markov model
            edges = list(graph.edges())
            vertices = list(graph.nodes)
            mm = MarkovModel()
            mm.add_nodes_from(vertices)
            mm.add_edges_from(edges)
            bm = mm.to_bayesian_model()

            return bm

