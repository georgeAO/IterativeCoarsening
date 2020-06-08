import networkx as nx
from itertools import combinations, product


def mcs(graph):
    '''
    Parameters
    ----------
    graph : A networkx graph object which contains an undirected graph

    Returns
    -------
    relabeled : A networkx graph object which renumbers G
    renumbering : A dictionary which maps the old numbering from G
    to the new numbering in H

    '''
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
    '''
    Parameters
    ----------
    graph : A networkx graph object which contains an undirected
    chordal graph

    Returns
    -------
    finalList : A list of cliques (list(set(int))) which satisfy the running
    intersection property
    '''
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
    '''
    Parameters
    ----------
    clique_list : A list of cliques (list(set(int))) which satisfy the running
    intersection property

    Returns
    -------
    sepList : A list of separators list(frozenset(int)) for the graph defined by cliqueList
    '''
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
    '''
    Parameters
    ----------
    graph : A networkx undirected graph
    sep_list : A list of separators list(set(int))

    Returns
    -------
    compList : A list of the connected components list(list(set(int))) corresponding to each separator.
    That is, compList[i] contains the connected components when s[i] is removed from G

    '''
    comp_list = list()
    for S in sep_list:
        copied = graph.copy()
        for v in S:
            copied.remove_node(v)
        comp_list.append(list(nx.algorithms.components.connected_components(copied)))

    return comp_list


def get_seps_for_cand_edges(comps, seps):
    '''
    For every candidate edge (u,v) (edge that is formed by to vertices in different separated components given a separator)
    obtain the list of separators of u and v in the (decomposable) undirected graph associated to comps and seps.

    :param comps: a list of connected components separated by each of the separators in the list seps, list(list(set(int)
    :param seps: a list of separators that separate the connected components in the list comps, list(set(int)
    :param n: the number of vertices in the (decomposable) undirected graph
    :return: A dictionary that maps every candidate edge, (a,b) where a and b are int and a<b, to its list of separators
    '''

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
    '''
    Parameters
    ----------
    sol : The solution obtained by Gurobi's stdgrb.lp_solve method. This array encodes which edges will
    be added to the graph. If sol[i] == 0, the edge is not added. If sol[i] == 1, the edge is added.
    dict_of_vars : A dictionary which maps the decision variables to their respective index in sol.

    Returns
    -------
    newEdges : A list(tuple()) which contains the new edges to add to the graph.

    '''
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
    '''
    Parameters
    ----------
    graph : A networkx graph G

    Returns
    -------
    edges : the edges of G as a list(tuple()) such that for every tuple (a,b) in edges
    we have a < b.

    '''
    edges = list(graph.edges())
    for index, edge in enumerate(edges):
        edges[index] = tuple(sorted(edge))
    return edges


def mkdg_check(graph):
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
    edgesG1 = get_sorted_edges(graph1)
    edgesG2 = get_sorted_edges(graph2)
    count = 0
    for edge in edgesG1:
        if edge in edgesG2:
            count += 1

    percentage = count / len(edgesG2)

    return percentage


def different_edges(graph1, graph2):
    edgesG1 = get_sorted_edges(graph1)
    edgesG2 = get_sorted_edges(graph2)
    count = 0

    for edge in edgesG1:
        if edge not in edgesG2:
            count += 1

    percentage = count / len(edgesG1)

    return percentage
