import numpy as np


def decode_MST(energies, leading_symbolic=0, labeled=True):
    """
    decode best parsing tree with MST algorithm.
    :param energies: energies: numpy 3D tensor
        energies of each edge. the shape is [num_labels, n_steps, n_steps],
        where the summy root is at index 0.
    :param leading_symbolic: int
        number of symbolic dependency labels leading in label alphabets)
    :return:
    """

    def find_cycle(par):
        added = np.zeros([length], np.bool)
        added[0] = True
        cycle = set()
        findcycle = False
        for i in range(1, length):
            if findcycle:
                break

            if added[i] or not curr_nodes[i]:
                continue

            # init cycle
            tmp_cycle = set()
            tmp_cycle.add(i)
            added[i] = True
            findcycle = True
            l = i

            while par[l] not in tmp_cycle:
                l = par[l]
                if added[l]:
                    findcycle = False
                    break
                added[l] = True
                tmp_cycle.add(l)

            if findcycle:
                lorg = l
                cycle.add(lorg)
                l = par[lorg]
                while l != lorg:
                    cycle.add(l)
                    l = par[l]
                break

        return findcycle, cycle

    def chuLiuEdmonds():
        par = np.zeros([length], dtype=np.int32)
        # create best graph
        par[0] = -1
        for i in range(1, length):
            # only interested at current nodes
            if curr_nodes[i]:
                max_score = score_matrix[0, i]
                par[i] = 0
                for j in range(1, length):
                    if j == i or not curr_nodes[j]:
                        continue

                    new_score = score_matrix[j, i]
                    if new_score > max_score:
                        max_score = new_score
                        par[i] = j

        # find a cycle
        findcycle, cycle = find_cycle(par)
        # no cycles, get all edges and return them.
        if not findcycle:
            final_edges[0] = -1
            for i in range(1, length):
                if not curr_nodes[i]:
                    continue

                pr = oldI[par[i], i]
                ch = oldO[par[i], i]
                final_edges[ch] = pr
            return

        cyc_len = len(cycle)
        cyc_weight = 0.0
        cyc_nodes = np.zeros([cyc_len], dtype=np.int32)
        id = 0
        for cyc_node in cycle:
            cyc_nodes[id] = cyc_node
            id += 1
            cyc_weight += score_matrix[par[cyc_node], cyc_node]

        rep = cyc_nodes[0]
        for i in range(length):
            if not curr_nodes[i] or i in cycle:
                continue

            max1 = float("-inf")
            wh1 = -1
            max2 = float("-inf")
            wh2 = -1

            for j in range(cyc_len):
                j1 = cyc_nodes[j]
                if score_matrix[j1, i] > max1:
                    max1 = score_matrix[j1, i]
                    wh1 = j1

                scr = cyc_weight + score_matrix[i, j1] - score_matrix[par[j1], j1]

                if scr > max2:
                    max2 = scr
                    wh2 = j1

            score_matrix[rep, i] = max1
            oldI[rep, i] = oldI[wh1, i]
            oldO[rep, i] = oldO[wh1, i]
            score_matrix[i, rep] = max2
            oldO[i, rep] = oldO[i, wh2]
            oldI[i, rep] = oldI[i, wh2]

        rep_cons = []
        for i in range(cyc_len):
            rep_cons.append(set())
            cyc_node = cyc_nodes[i]
            for cc in reps[cyc_node]:
                rep_cons[i].add(cc)

        for i in range(1, cyc_len):
            cyc_node = cyc_nodes[i]
            curr_nodes[cyc_node] = False
            for cc in reps[cyc_node]:
                reps[rep].add(cc)

        chuLiuEdmonds()

        # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
        found = False
        wh = -1
        for i in range(cyc_len):
            for repc in rep_cons[i]:
                if repc in final_edges:
                    wh = cyc_nodes[i]
                    found = True
                    break
            if found:
                break

        l = par[wh]
        while l != wh:
            ch = oldO[par[l], l]
            pr = oldI[par[l], l]
            final_edges[ch] = pr
            l = par[l]

    if labeled:
        assert energies.ndim == 3, 'dimension of energies is not equal to 3'
    else:
        assert energies.ndim == 2, 'dimension of energies is not equal to 2'
    
    input_shape = energies.shape
    length = input_shape[1]

    # calc real energies matrix shape = [length, length, num_labels - #symbolic] (remove the label for symbolic labels).
    if labeled:
        energies = energies[leading_symbolic:, :, :]
        # get best label for each edge.
        label_id_matrix = energies.argmax(axis=0) + leading_symbolic
        energies = energies.max(axis=0)
    else:
        label_id_matrix = None
    # get original score matrix
    orig_score_matrix = energies
    # initialize score matrix to original score matrix
    score_matrix = np.array(orig_score_matrix, copy=True)

    oldI = np.zeros([length, length], dtype=np.int32)
    oldO = np.zeros([length, length], dtype=np.int32)
    curr_nodes = np.zeros([length], dtype=np.bool)
    reps = []

    for s in range(length):
        orig_score_matrix[s, s] = 0.0
        score_matrix[s, s] = 0.0
        curr_nodes[s] = True
        reps.append(set())
        reps[s].add(s)
        for t in range(s + 1, length):
            oldI[s, t] = s
            oldO[s, t] = t

            oldI[t, s] = t
            oldO[t, s] = s

    final_edges = dict()
    chuLiuEdmonds()
    par = np.zeros([length], np.int32)
    if labeled:
        label = np.ones([length], np.int32)
        label[0] = 0
    else:
        label = None

    for ch, pr in final_edges.items():
        par[ch] = pr
        if labeled and ch != 0:
            label[ch] = label_id_matrix[pr, ch]

    par[0] = 0

    return par.tolist()[1:], label.tolist()[1:]


def decode_GGDP_projective(energies, leading_symbolic=0, labeled=True):
    if labeled:
        assert energies.ndim == 3, 'dimension of energies is not equal to 3'
    else:
        assert energies.ndim == 2, 'dimension of energies is not equal to 2'
    
    input_shape = energies.shape
    length = input_shape[1]

    par = np.zeros([length], np.int32)
    if labeled:
        label = np.ones([length], np.int32)
        label[0] = 0
    else:
        label = None

    # calc real energies matrix shape = [length, length, num_labels - #symbolic] (remove the label for symbolic labels).
    if labeled:
        energies = energies[leading_symbolic:, :, :]
        # get best label for each edge.
        label_id_matrix = energies.argmax(axis=0) + leading_symbolic
        energies = energies.max(axis=0)
    else:
        label_id_matrix = None
    # get original score matrix
    orig_score_matrix = energies
    # initialize score matrix to original score matrix
    score_matrix = np.array(orig_score_matrix, copy=True)

    # because ROOT will not have head
    score_matrix[:,0] = float('-inf')

    pending_list = [idx for idx in range(length)]

    while len(pending_list) > 1:
        valid_head, valid_dep, valid_score = None, None, float('-inf')
        selected_head, selected_dep = None, None

        for idx in range(len(pending_list) - 1):
            # AttachLeft or AttachRight
            for op in range(2):
                hi, di = idx + (1 - op), idx + op
            
                head_id = pending_list[hi]
                dep_id = pending_list[di]

                if valid_score < score_matrix[head_id, dep_id]:
                    valid_score = score_matrix[head_id, dep_id]
                    valid_head, valid_dep = head_id, dep_id
                    selected_head, selected_dep = hi, di

        assert valid_head is not None, 'Fatal Error!'

        par[valid_dep] = valid_head

        if labeled:
            label[valid_dep] = label_id_matrix[valid_head, valid_dep]

        del pending_list[selected_dep]

    par[0] = 0

    return par.tolist()[1:], label.tolist()[1:]


def decode_GGDP_nonprojective(energies, order_scores, leading_symbolic=0, labeled=True):
    if labeled:
        assert energies.ndim == 3, 'dimension of energies is not equal to 3'
    else:
        assert energies.ndim == 2, 'dimension of energies is not equal to 2'
    
    input_shape = energies.shape
    length = input_shape[1]

    par = np.zeros([length], np.int32)
    if labeled:
        label = np.ones([length], np.int32)
        label[0] = 0
    else:
        label = None

    # calc real energies matrix shape = [length, length, num_labels - #symbolic] (remove the label for symbolic labels).
    if labeled:
        energies = energies[leading_symbolic:, :, :]
        # get best label for each edge.
        label_id_matrix = energies.argmax(axis=0) + leading_symbolic
        energies = energies.max(axis=0)
    else:
        label_id_matrix = None

    # get original score matrix
    orig_score_matrix = energies
    # initialize score matrix to original score matrix
    score_matrix = np.array(orig_score_matrix, copy=True)

    # because ROOT will not have head
    score_matrix[:,0] = float('-inf')
    
    # make root the last position
    order_scores[0] = float('-inf')

    pending_list = list(np.argsort(-order_scores))

    while len(pending_list) > 1:
        valid_head, valid_dep, valid_score = None, None, float('-inf')
        selected_head, selected_dep = None, None

        for idx in range(1, len(pending_list)):

            hi, di = idx, 0

            head_id = pending_list[hi]
            dep_id = pending_list[di]

            if valid_score < score_matrix[head_id, dep_id]:
                valid_score = score_matrix[head_id, dep_id]
                valid_head, valid_dep = head_id, dep_id
                selected_head, selected_dep = hi, di

        assert valid_head is not None, 'Fatal Error!'

        par[valid_dep] = valid_head

        if labeled:
            label[valid_dep] = label_id_matrix[valid_head, valid_dep]

        del pending_list[selected_dep]

    par[0] = 0

    return par.tolist()[1:], label.tolist()[1:]





