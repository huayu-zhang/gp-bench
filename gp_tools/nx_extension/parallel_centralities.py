import multiprocessing as mp
import itertools
import networkx as nx


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, n_jobs=None):
    """Parallel betweenness centrality  function"""
    p = mp.Pool(n_jobs)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_source,
        zip([G] * num_chunks, [True] * num_chunks, [None] * num_chunks, node_chunks),
    )

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c


counter = mp.Value('i', 0)


def _closeness_centrality_wrap(G, source):
    global counter
    with counter.get_lock():
        counter.value += 1
        print('Fnished (%s/%s)' % (counter.value, len(G)), end='\r')
    return nx.closeness_centrality(G, source)


def closeness_centrality_parallel(G, n_jobs=None):
    map_gen = ((G, node) for node in G.nodes)

    pool = mp.Pool(n_jobs)
    list_closeness = pool.starmap(_closeness_centrality_wrap, map_gen)
    pool.close()

    closeness = {node: value for node, value in zip(G.nodes, list_closeness)}

    return closeness
