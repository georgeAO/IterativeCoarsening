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
from cliquetree import CliqueTree
from gurobipy import GRB
from itertools import combinations


class DecomposableModel:
    """ A class for learning decomposable models from coarsened kDGs.
    Attributes:
        alpha (int): the equivalent sample size of the Dirichlet uniform prior.
        data (np.array): An np.array which holds the data set.
        num_vars (int): The number of variables.
        data_frame (pd.DataFrame): The data frame version of data.
        var_states (list): For each i in range(numvars), varstates[i] is the number of variable states observed.
        bdeu (pgmpy.BDeuScore): A BDeuScore object with uniform prior alpha used to score networks.
        mst (nx.Graph): A copy of the maximal spanning tree which is learned from the BDeu score of the data
        undirected (nx.Graph): A networkx graph which stores the learned undirected models.
        directed (pgmpy.BayesianModel): A pgmpy BayesianModel which stores the learned directed model via minimum I-map.
    """
    def __init__(self, fileName, alpha=1):
        """ Build the empty graph model with n=data.shape[1] vertices. Set other attributes for use in model learning.
        Args:
            :param fileName (string): The path of the .csv file containing the data.
            :param alpha (int): the equivalent sample size of the Dirichlet uniform prior.
        """

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

        self.maxtree = self.undirected.copy()

        self.directed = BayesianModel()

    def mst(self):
        """Builds the maximum spanning tree for the given data and alpha. The results is kept in the undirected
        attribute.
        """
        complete_graph = nx.Graph()
        lexicographic_edges = list(combinations(list(range(self.num_vars)), 2))
        weight_list = self.get_weight_list(lexicographic_edges)

        for index, edge in enumerate(lexicographic_edges):
            complete_graph.add_edge(edge[0], edge[1], weight=-1 * weight_list[index])

        edges = list(nx.algorithms.tree.minimum_spanning_edges(complete_graph, algorithm='kruskal', data=False))
        self.maxtree.add_edges_from(edges)

        if nx.classes.function.is_empty(self.directed):
            self.undirected = self.maxtree.copy()


    def get_score(self, V):
        """ The score function.
        In the case of alpha=0, it computes the entropy of a given set of variables.
        In the case of alpha>0, it computes the BDeu score with equivalent sample size equal to alpha associated to
        a set of variables: BDeu(V)= sum_v ln(gamma(N_v + alpha/r_V))- ln(gamma(alpha/r_V) where N_v denotes the number
        of cases of the dataset in which variable V takes the value v, and r_V denotes the cardinality (the number of
        different values) of the set of variables V.
        Remarks: this method only works for discrete random variables.
        :param V (list): A subset of variables given by their indices, list(int).
        :return scores (np.array): The BDeu scores of variables V given the data set D.
        """

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
        """ Return the scores for edges which could be added to the empty graph.
        :param edges (list(tuples)): The list of edges to consider adding to the empty graph.
        :return weight_list (list):  The list of weights corresponding to each edge in edges.
        """
        weight_list = []
        for edge in edges:
            weight_list.append(self.get_score([edge[0]]) + self.get_score([edge[1]]) - self.get_score(list(edge))
                              - self.get_score([]))
        return weight_list

    def get_objective(self, dict_of_vars):
        """ In the ILP model formulation of https://opt-ml.org/oldopt/papers/OPT2015_paper_36, this defines the vector
        w_{u,v|S}. That is, each element is the BDeu score related to adding edge (u,v) given separator S.
        :param dict_of_vars (dictionary): A dictionary which maps each candidate {u,v|S} to an index i in w.
        :return w (np.array): The vector of weights given the dictionary elements. That is, w[i] = score({u,v|S}).
        """
        w = np.zeros(len(dict_of_vars.keys()))
        for key in dict_of_vars.keys():
            to_scoreUVS = list({key[0][0], key[0][1]}.union(key[1]))
            to_scoreUS = list({key[0][0]}.union(key[1]))
            to_scoreVS = list({key[0][1]}.union(key[1]))
            to_scoreS = list(key[1])

            w[dict_of_vars[key]] = self.get_score(to_scoreUS) + self.get_score(to_scoreVS) - \
                                   self.get_score(to_scoreUVS) - self.get_score(to_scoreS)
        return w

    def learn(self, k_max=np.inf, l_max=np.inf, max_time=1000.0, max_time_gurobi=500.0, maximal=False):
        """An algorithm which performs the learning of kDGs (or MkDGs) by the coarsening procedure proposed in
        https://opt-ml.org/oldopt/papers/OPT2015_paper_36. The learned graph is placed in the undirected attribute
        :param k_max (int): The maximal clique size of the learned network.
        :param l_max (int): The maximal length of an edge which may be considered for addition to the model.
        :param max_time (float): The maximum running time of the coarsening algorithm.
        :param max_time_gurobi (float): The maximum running time for each call to gurobi
        :param maximal (bool): Whether to learn a kDG (False) or an MkDG (True).
        Remarks: The maximal running time is only considers terminating the learning process after each
        coarsening step. Therefore, the actual running time may be longer than what is specified by max_time
        """
        if nx.classes.function.is_empty(self.maxtree):
            self.mst()

        current_model = self.maxtree
        sol = [1]
        current_max_clique = 2
        start = time.time()
        end = 0
        while np.linalg.norm(sol, ord=1) > 0.01 and (end - start) < max_time:
            edge_list = ch.get_sorted_edges(current_model)
            cliques = ch.chain_of_cliques(current_model)

            if not maximal:
                for clique in cliques:
                    if len(clique) > current_max_clique:
                        current_max_clique = len(clique)

            if current_max_clique == k_max:
                break

            sep_list = ch.get_separators(cliques)
            comps = ch.get_comp_list(current_model, sep_list)
            sep_to_comps = dict(zip(sep_list, comps))
            e_to_seps = ch.get_seps_for_cand_edges(comps, sep_list)
            list_vars = co.generate_dec_variables(e_to_seps, l_max)
            dict_of_vars = {uvS: ind for ind, uvS in enumerate(list_vars)}
            c = self.get_objective(dict_of_vars)

            m = gp.Model("iterate" + str(current_max_clique))
            x = m.addMVar(shape=len(c), vtype=GRB.BINARY, name="x")
            m.setObjective(c @ x, GRB.MAXIMIZE)
            m.Params.MIPFocus = 1
            m.Params.timeLimit = max_time_gurobi

            A1, b1 = co.type1(e_to_seps, dict_of_vars)
            m.addConstr(A1 @ x <= b1, name="c1")

            A2u, b2u, A2v, b2v = co.type2(e_to_seps, dict_of_vars, edge_list)
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

    def to_bn(self, use_mst=False):
        """ Given the undirected graph, return the directed model corresponding to the minimal I-map. The directed model is
        placed in the directed attribute.
        Remarks: This method supports models which are not connected.
        :param use_mst (boolean): To direct a prelearned model in the directed attribute, direct_mst=False. To obtain
            the directed model from the maximum spanning tree learned by self.mst, use direct_mst=True.
        """

        # Generate connected components
        if use_mst:
            edges = list(self.maxtree.edges())
            vertices = list(self.maxtree.nodes)
            mm = MarkovModel()
            mm.add_nodes_from(vertices)
            mm.add_edges_from(edges)
            bm = mm.to_bayesian_model()

            self.directed = bm

        else:
            connected = list(nx.algorithms.components.connected_components(self.undirected))
            # Hold the edges of each connected component as a list(list())
            connected_comps = list()
            # If the graph is not completely connected
            if len(connected) > 1:
                self.directed = BayesianModel()
                for comp in connected:
                    # If the connected component is not a single vertex
                    if len(comp) > 1:
                        edges_to_add = list()
                        temp = MarkovModel()
                        for vert1 in comp:
                            neighbours = list(self.undirected.neighbors(vert1))
                            for vert2 in neighbours:
                                if not ((vert1, vert2) in edges_to_add) and not ((vert2, vert1) in edges_to_add):
                                    edges_to_add.append((vert1, vert2))

                        temp.add_nodes_from(comp)
                        temp.add_edges_from(edges_to_add)
                        temp_bn = temp.to_bayesian_model()
                        connected_comps.append(temp_bn)
                    else:
                        self.directed.add_nodes_from(comp)

                for bn in connected_comps:
                    self.directed.add_nodes_from(list(bn.nodes))
                    self.directed.add_edges_from(list(bn.edges))

            else:
                # If the graph is completely connected, just add all edges to markov model
                edges = list(self.undirected.edges())
                vertices = list(self.undirected.nodes)
                mm = MarkovModel()
                mm.add_nodes_from(vertices)
                mm.add_edges_from(edges)
                bm = mm.to_bayesian_model()

                self.directed = bm

    def fic(self):
        """It obtain the maximal set of edges of length 1 with the maximum total weight that can be added to the current
        structure. This is equivalent to adding edges of length 1 with the ILP formulation with the equality condition in
        type 3 constraints


        :return: The set of edges due to the different separators that can be added, list(((u,v),S) where S is a
        frozenset(int) and u< v
        """
        current_model = self.undirected
        cliques = ch.chain_of_cliques(current_model)
        sep_list = ch.get_separators(cliques)
        comps = ch.get_comp_list(current_model, sep_list)
        sep_to_comp = dict(zip(sep_list, comps))
        edges = list()
        for S in sep_to_comp:
            # Compute the weights an select the best edges among different connected components in the mantle
            n = len(sep_to_comp[S])
            E = np.ndarray(shape=(n, n, 2), dtype=int)
            W = np.zeros(shape=(n, n), dtype=float)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    wMax = -np.inf
                    uMax = -1
                    vMax = -1
                    for u in sep_to_comp[S][i]:
                        for v in sep_to_comp[S][j]:
                            # Replace this function by the function in which you compute the score
                            w = self.get_score([u] + list(S)) + self.get_score([v] + list(S)) \
                                - self.get_score([u, v] + list(S)) - self.get_score(list(S))
                            if (w > wMax):
                                wMax = w
                                uMax = u
                                vMax = v
                    E[i, j] = [uMax, vMax]
                    W[i, j] = wMax
                    E[j, i] = [uMax, vMax]
                    W[j, i] = wMax

            # Construct the maximum weighted spanning tree (Prim's algorithm with adjacency lists) coarser than the given forest
            unvisited = list(range(1, n))
            maxU = np.zeros(n).astype(int)
            maxW = W[0, :]

            for l in range(n - 1):
                ind = np.argmax(maxW[unvisited])
                v = unvisited[ind]
                u = maxU[v]

                if E[u, v][0] < E[u, v][1]:
                    e = (E[u, v][0], E[u, v][1])
                else:
                    e = (E[u, v][1], E[u, v][0])
                edges.append((e, S))
                del unvisited[ind]
                for w in unvisited:
                    if (maxW[w] < W[v, w]):
                        maxW[w] = W[v, w]
                        maxU[w] = v

        to_add = [item[0] for item in edges]

        return to_add

    def coarsen(self):
        """Coarsen the current model in self.directed by using the procedure described in
            "Efficient approximation of probability distributions with k-order decomposable models". This approach
            is called Forced Iterative Coarsening (FIC) in "Learning decomposable models by coarsening".
        """

        edges_to_add = self.fic()
        self.undirected.add_edges_from(edges_to_add)

    def greedy_learn(self, k_max=np.inf, time_limit=np.inf):
        """An algorithm for learning kDGs using a greedy hill-climbing approach. The code is influenced by the design
        of the hill climbing approach used in the pgmpy package
        (https://pgmpy.org/_modules/pgmpy/estimators/HillClimbSearch.html). However, we only consider edge additions
        which maintain chordality. The learned models are placed in the undirected and directed attributes.
        :param k_max (int): The maximal clique size of the graph to learn
        :param time_limit: The time limit of the search
        """
        # build the mst spanning tree
        initial_graph = CliqueTree()
        start = time.time()
        #Generate MST
        if nx.classes.function.is_empty(self.maxtree):
            self.mst()

        for edge in list(self.get_model_mst().edges()):
            initial_graph.add_edge(edge[0], edge[1])

        # the best bn learned so far is the MST
        self.to_bn(use_mst=True)
        best_bn = self.get_model_directed()
        # Flag which controls when the algorithm finishes
        add_edge = True
        if k_max > 2:
            while add_edge:
                running_time = time.time() - start
                best_score_delta = (0, None)
                if running_time > time_limit:
                    break
                current_bn = best_bn.copy()
                add_edge = False
                # make a list of all possible edges that could be added to the graph
                diff = initial_graph.insertable
                for edge in diff:
                    running_time = time.time() - start
                    if running_time > time_limit:
                        break
                    # add an edge to the graph
                    initial_graph.add_edge(edge[0], edge[1])
                    cliques = initial_graph.nodes_in_clique
                    # check if the graph is chordal and has the right clique number
                    if DecomposableModel.get_clique_num(cliques) <= k_max:
                        # get the score of the new graph
                        greedy_bn, greedy_score_delta = DecomposableModel.add_edge_to_bn(edge, current_bn.copy(), self.bdeu)
                        # is the score better than previous graphs?
                        if greedy_score_delta > best_score_delta[0]:
                            # If we add can add an edge, the algorithm should continue looking for more edges
                            add_edge = True
                            best_score_delta = (greedy_score_delta, edge)
                            best_bn = greedy_bn

                    # after we check the candidate edge, remove it from the graph and keep looking
                    initial_graph.remove_edge(edge[0], edge[1])

                # make the new initial graph the best graph from the previous search
                if add_edge:
                    initial_graph.add_edge(best_score_delta[1][0], best_score_delta[1][1])

        self.undirected = initial_graph.G
        self.directed = best_bn

    @staticmethod
    def get_clique_num(cliques):
        """ A static method for greedy_learn which return the maximal clique size of a model.
        :param cliques (dict): The dictionary given by the nodes_in_clique attribute of CliqueTree. The cliques are the
            dictionary keys.
        :return clique_number (int): The number of vertices in the largest clique.
        """
        clique_number = 0
        for key in cliques.keys():
            if len(cliques[key]) > clique_number:
                clique_number = len(cliques[key])

        return clique_number

    @staticmethod
    def add_edge_to_bn(edge, bn, BDeu):
        """ A static method for greedy_learn which computes the local score delta for adding an edge. The correct
            direction of the edge is also determined for the directed model.
        :param edge tuple(int): The edge (undirected) to consider adding.
        :param bn (pgmpy.BayesianModel): The current bn model we wish to add edge to.
        :param BDeu (pgmpy.BDeuScore): The BDeu score object which calculate the local scores.
        :return bn (pgmpy.BayesianModel): The bn provided by the user with edge added in the correct orientation
        :return score_delta (float): The best local score delta for adding the edge to the current bn.
        """
        try:
            local_score = BDeu.local_score
            bn.add_edge(edge[0], edge[1])
            new_parents = bn.get_parents(edge[1])
            old_parents = list(set(new_parents) - {edge[0]})
            score_delta1 = local_score(edge[1], new_parents) - local_score(edge[1], old_parents)

            bn.remove_edge(edge[0], edge[1])

            bn.add_edge(edge[1], edge[0])
            new_parents = bn.get_parents(edge[0])
            old_parents = list(set(new_parents) - {edge[1]})
            score_delta2 = local_score(edge[0], new_parents) - local_score(edge[0], old_parents)

            bn.remove_edge(edge[1], edge[0])

            if score_delta1 > score_delta2:
                bn.add_edge(edge[0], edge[1])
                return bn, score_delta1
            else:
                bn.add_edge(edge[1], edge[0])
                return bn, score_delta2
        except ValueError:
            try:
                bn.add_edge(edge[0], edge[1])
                new_parents = bn.get_parents(edge[1])
                old_parents = list(set(new_parents) - {edge[0]})
                score_delta1 = local_score(edge[1], new_parents) - local_score(edge[1], old_parents)

                return bn, score_delta1
            except ValueError:
                bn.add_edge(edge[1], edge[0])
                new_parents = bn.get_parents(edge[0])
                old_parents = list(set(new_parents) - {edge[1]})
                score_delta2 = local_score(edge[0], new_parents) - local_score(edge[0], old_parents)
                return bn, score_delta2

    def get_model_directed(self):
        return self.directed.copy()

    def get_model_undirected(self):
        return self.undirected.copy()

    def get_model_mst(self):
        return self.maxtree.copy()

    def get_score_function(self):
        return self.bdeu


