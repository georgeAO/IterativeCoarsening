import numpy as np
import scipy.sparse as sp
from itertools import combinations, chain


def powerset(iterable):
    '''
    A customized version of powerset which produces
    all subsets of iterable of length 2 to length(iterable)
    i.e. powerset([1,2,3]) --> (1,2) (1,3) (2,3) (1,2,3)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2, len(s) + 1))


def is_it_possible(e, s, e_to_seps):
    '''
    Checks if it is possible to add an edge due to a separator given the current set of minimal separators

    :param e: candidate edge, tuple of ints (u,v) where u<v
    :param s: sepsrator
    :param e_to_seps: Dictionary from edges to list of separators
    :return: true if it is possible to add the edge using the minimal separators
    '''

    for u in e:
        num = 0
        seps = set()
        # Count the edges that need to be added and the number of separators that can be involved in their addition
        for w in s:

            if w < u:
                uw = (w, u)
            else:
                uw = (u, w)

            if uw in e_to_seps:
                num += 1
                seps.update(e_to_seps[uw])

        if num > len(seps):
            return False

    return True


def generateDecVariables(e_to_seps, l):
    '''
    Obtains the decision variables X_{u,v|S} codified in terms of the edge and the separator, ((u,v),S)

    Observation: only the edges due to a separator that can be added given the current list of minimal separators
    are generated

    :param e_to_seps: A dictionary that maps every candidate edge, (a,b) where a and b are int and a<b, to its list of
    separators (see function getSepsForCandEdges(comps,seps,n))
    :param l: the constraint to the maximum length of an edge to be considered a variable of the problem (see PGM 2018)
    :return: A list with the decision variables X_{u,v|S}, list(tuples) where every tuple has the form ((u,v),S), with
    u and v are ints and u<v, and S is a set of integers
    '''

    dec_vars = list()
    for e in e_to_seps.keys():
        if len(e_to_seps[e]) <= l:
            for S in e_to_seps[e]:
                if is_it_possible(e, S, e_to_seps):
                    dec_vars.append((e, S))

    return dec_vars


def type1(e_to_seps, dict_of_vars):
    '''
    Obtains the type 1 constraints of the ILP formulation: this constraints guarantee that each edge can be added once
    at most, due to a single separator

    Constr1:
    for each (u,v) sum_{S in seps(u,v)} X_u,v|S <=1
    where seps(u,v) denotes all the separators of u and v in G
    Format:
    A constraint is codified as [e,S1,...,Sm] where {S1,...,Sm}= seps(u,v)


    :param e_to_seps: A dictionary that maps every candidate edge, (a,b) where a and b are int and a<b, to its list of
    separators (see function getSepsForCandEdges(comps,seps,n))
    :param l: the constraint to the maximum length of an edge to be considered a variable of the problem (see PGM 2018)
    :return: Returns the tipe 1 constraints of the ILP formulation (see the comments above)
    '''
    decVariables = dict_of_vars.keys()
    constr1 = list()
    for e in e_to_seps.keys():
        const = [e]
        for S in e_to_seps[e]:
            const.append(S)

        constr1.append(const)

    sizeRows = 0
    sizeColumns = len(dict_of_vars.keys())
    for key in e_to_seps.keys():
        if len(e_to_seps[key]) >= 2:
            sizeRows += 1
    A1 = sp.lil_matrix((sizeRows, sizeColumns))
    b1 = np.ones((sizeRows, ))
    count = 0
    for c in constr1:
        if len(c) > 2:
            for i in range(1, len(c)):
                if (c[0], c[i]) in decVariables:
                    A1[count, dict_of_vars[(c[0], c[i])]] = 1
            count += 1

    return A1, b1


def type2(e_to_seps, dict_of_vars, edge_list, k):
    dec_vars = dict_of_vars.keys()
    n = len(dec_vars)
    A2u = sp.lil_matrix((n, n))
    b2u = np.zeros((n, ))
    A2v = sp.lil_matrix((n, n))
    b2v = np.zeros((n, ))
    count = 0
    for dec_var in dec_vars:
        uv = dec_var[0]
        u = uv[0]
        v = uv[1]
        for s in dec_var[1]:
            for R in e_to_seps[uv]:
                pair1 = tuple(sorted((u, s)))
                pair2 = tuple(sorted((v, s)))
                if (pair1, R) in dec_vars:
                    A2u[count, dict_of_vars[(pair1, R)]] = -1
                if (pair2, R) in dec_vars:
                    A2v[count, dict_of_vars[(pair2, R)]] = -1
            if pair1 in edge_list:
                b2u[count] += 1
            if pair2 in edge_list:
                b2v[count] += 1
        A2u[count, dict_of_vars[dec_var]] = k - 1
        A2v[count, dict_of_vars[dec_var]] = k - 1
        count += 1
    return A2u, b2u, A2v, b2v


def type3(sep_to_comps, dict_of_vars):
    dec_vars = dict_of_vars.keys()
    A3ineq = list()
    b3ineq = list()
    A3eq = list()
    b3eq = list()
    for s in sep_to_comps.keys():
        comps = sep_to_comps[s]
        p_set = list(powerset(comps))
        for p in p_set:
            hold_ineq = list()
            hold_eq = list()
            add_ineq = False
            if not (len(p) == len(comps)):
                add_ineq = True
                combos = list(combinations(p, 2))
                for pair in combos:
                    for u in pair[0]:
                        for v in pair[1]:
                            uv = tuple(sorted([u, v]))
                            if (uv, s) in dec_vars:
                                hold_ineq.append(dict_of_vars[(uv, s)])
            else:
                combos = list(combinations(p, 2))
                for pair in combos:
                    for u in pair[0]:
                        for v in pair[1]:
                            uv = tuple(sorted([u, v]))
                            if (uv, s) in dec_vars:
                                hold_eq.append(dict_of_vars[(uv, s)])
            if add_ineq:
                A3ineq.append(hold_ineq)
                b3ineq.append(len(p) - 1)
            else:
                A3eq.append(hold_eq)
                b3eq.append(len(p) - 1)

    A3 = sp.lil_matrix((len(A3ineq) + len(A3eq), len(dict_of_vars.keys())))
    b3 = np.zeros((len(b3ineq) + len(b3eq), ))
    row = 0
    for item in A3eq:
        for vals in item:
            A3[row, vals] = 1
        b3[row] = b3eq[row]
        row += 1
    hold = row
    for item in A3ineq:
        for vals in item:
            A3[row, vals] = 1
        b3[row] = b3ineq[row - hold]
        row += 1
    return A3, b3, len(b3eq)


def generateConstraints(EtoSeps, dictOfVars, SepToComps, EdgeList, k):
    A1, b1 = type1(EtoSeps, dictOfVars)
    A2u, b2u, A2v, b2v = type2(EtoSeps, dictOfVars, EdgeList, k)
    A3, b3, r = type3(SepToComps, dictOfVars)

    A = np.vstack((A3, A1, A2u, A2v))
    b = np.vstack((b3, b1, b2u, b2v))

    return A, b.reshape((b.shape[0],)), r
