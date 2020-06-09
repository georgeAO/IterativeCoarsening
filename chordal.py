import networkx as nx
from itertools import combinations, product


def mcs(graph):
    """ Perform a maximum cardinality search and relabeling.
    :param graph (nx.Graph): The chordal graph to label.

    :return relabeled (nx.Graph): The relabeled graph.
    :return renumbering (dictionary): The dictionary which maps the old numbering of graph to the new numbering of
        renumbering.
    """
    count = 0
    vertices = set(graph.nodes)
    renumbering = {}
    renumbering[vertices.pop()] = count

    while vertices:
        max_neighbors = 0
        next_vertex = 0
        for vertex in vertices:
            neighbors = len(set(graph.neighbors(vertex)).intersection(set(renumbering.keys())))
            if neighbors > max_neighbors:
                max_neighbors = neighbors
                next_vertex = vertex
        count += 1
        renumbering[next_vertex] = count
        vertices.remove(next_vertex)

    relabeled = nx.relabel.relabel_nodes(graph, renumbering)

    return relabeled, renumbering


def chain_of_cliques(graph):
    """Given a chordal graph, generate a list of cliques which satisfies the running intersection property.

    :param graph (nx.Graph) : The chordal graph.

    :return final_list (list(set(int): The list of sets of vertices which satisfy the running intersection property.
    """

    relabeled, renumbering = mcs(graph)
    inv_renumbering = {v: k for k, v in renumbering.items()}
    cliques = list(nx.algorithms.clique.find_cliques(relabeled))
    clique_list = []
    for clique in cliques:
        clique_list.append((max(clique), clique))
    clique_list.sort()

    final_list = []
    for item in clique_list:
        clique = item[1]
        relabeled_clique = []
        for vertex in clique:
            relabeled_clique.append(inv_renumbering[vertex])
        final_list.append(set(relabeled_clique))

    return final_list


def get_separators(clique_list, unique=True):
    """
    :param clique_list list(set(int)): A list of cliques satisfying the running intersection property.
    :param unique (bool): Whether the returned list of separators should have no duplicates (True)
        or allow duplicates (False).

    :return sep_list (list(int)): The list of separators.
    """
    seen = clique_list.pop(0)
    sep_list = []

    for clique in clique_list:
        sep = clique.intersection(seen)
        seen = seen.union(clique)
        sep_list.append(frozenset(sep))
    if unique:
        return list(set(sep_list))
    else:
        return sep_list


def get_comp_list(graph, sep_list):
    """Given the chordal graph and the list of separators, obtain the list of connected components.

    :param graph (nx.Graph): The chordal graph.
    :param sep_list (list(set(int)): The list of separators for graph.

    :return comp_list (list(list(int)): The list of connected components.
    """

    comp_list = list()
    for S in sep_list:
        copied = graph.copy()
        for v in S:
            copied.remove_node(v)
        comp_list.append(list(nx.algorithms.components.connected_components(copied)))

    return comp_list


def get_seps_for_cand_edges(comps, seps):
    """For every candidate edge (u,v) (edge that is formed by to vertices in different separated components
        given a separator) obtain the list of separators of u and v in the (decomposable) undirected
        graph associated to comps and seps.

    :param comps (list(list(int))): a list of connected components separated by each of the separators in the list seps.
    :param seps (list(set(int))): a list of separators that separate the connected components in the list comps.

    :return: A dictionary that maps every candidate edge, (a,b) where a and b are int and a<b, to its list of separators.
    """

    eto_seps = dict()
    for i in range(len(comps)):
        s = comps[i]
        combos = combinations(s, 2)
        temp = []
        for c in combos:
            temp = list(product(*c))
            for t in temp:
                sort_t = tuple(sorted(t))
                if sort_t in eto_seps:
                    eto_seps[sort_t].append(frozenset(sorted(seps[i])))
                else:
                    eto_seps[sort_t] = [frozenset(sorted(seps[i]))]

    return eto_seps


def update_graph(sol, dict_of_vars):
    """
    :param sol (list(int)): The solution obtained by Gurobi. This array encodes which edges will
    be added to the graph. If sol[i] == 0, the edge is not added. If sol[i] == 1, the edge is added.
    :param dict_of_vars (dictionary): A dictionary which maps each candidate {u,v|S} to an index i in sol.


    :return newEdges (list(tuple(int))): The list of new edges to add.

    """
    index_dict = {}
    new_edges = []
    for i in range(0, len(sol)):
        index_dict[i] = sol[i]
    for key in dict_of_vars:
        hold = dict_of_vars[key]
        if round(index_dict[hold]) == 1:
            new_edges.append(key[0])

    return new_edges


def get_sorted_edges(graph):
    """Given a graph object, return the list of edges but ensures that for each (u,v) in Edges(graph), we have u < v.

    :param graph (nx.graph) : The graph which will yield the edge list.

    :return edges (list(tuple(int)) : the edges of graph as a list(tuple()) such that for every tuple (u,v) in edges
    we have u < v.

    """
    edges = list(graph.edges())
    for index, edge in enumerate(edges):
        edges[index] = tuple(sorted(edge))
    return edges


def is_mkdg(graph):
    """A function which checks if a graph object is an MkDG. Executes print statements to describe current state of
    the graph (graph is chordal, not MkDG OR graph is not chordal).

    :param graph (nx.Graph): The graph we wish to check.

    :return (bool):  True if the graph is an MKDG, False otherwise.
    """

    if nx.algorithms.chordal.is_chordal(graph):
        cliques = list(nx.algorithms.clique.find_cliques(graph))
        if len(cliques) == graph.number_of_nodes() - len(cliques[0]) + 1:
            return True
        else:
            print("Graph is chordal, but not MKDG")
            return False
    else:
        print("Graph is not chordal")
        return False


def same_edges(graph1, graph2):
    """Compare (as a percentage) the number of edges in graph1 which are also contained in graph2. The percentage is
        calculated as |edges(G1).intersection(edges(G2))|/|edges(G2)|

    :param graph1 (nx.Graph): The graph we will compare.
    :param graph2 (nx.Graph): The graph we will compare to.

    :return percentage (float): The percentage as calculated in the above statement.
    """

    edgesG1 = get_sorted_edges(graph1)
    edgesG2 = get_sorted_edges(graph2)
    count = 0
    for edge in edgesG1:
        if edge in edgesG2:
            count += 1

    percentage = count / len(edgesG2)

    return percentage


def different_edges(graph1, graph2):
    """Compare (as a percentage) the number of edges in graph1 which are not contained in graph2. The percentage is
            calculated as |edges(G1).setminus(edges(G2))|/|edges(G1)|

        :param graph1 (nx.Graph): The graph we will compare.
        :param graph2 (nx.Graph): The graph we will compare to.

        :return percentage (float): The percentage as calculated in the above statement.
        """
    edgesG1 = get_sorted_edges(graph1)
    edgesG2 = get_sorted_edges(graph2)
    count = 0

    for edge in edgesG1:
        if edge not in edgesG2:
            count += 1

    percentage = count / len(edgesG1)

    return percentage
