import torch as th


def pad_edge_output(graph, edge_feats, max_degree):

    out_edge_feats = edge_feats.size(-1)
    degrees_per_dst = graph.in_degrees().tolist()

    edge_feats_per_dst = th.split(edge_feats, split_size_or_sections=degrees_per_dst, dim=0)
    pad_zeros = [th.zeros(max_degree - d, out_edge_feats, dtype=th.float, device=edge_feats.device) for d in degrees_per_dst]

    # print(f"pad_zeros = {pad_zeros}")
    padded_edge_feats = []
    for dst_nid, e_feats in enumerate(edge_feats_per_dst):
        # print(f"e_feats.size() = {e_feats.size()}, pad_zeros[dst_nid].size() = {pad_zeros[dst_nid].size()}")

        padded_edge_feats.append(
            th.cat([e_feats, pad_zeros[dst_nid]], dim=0).flatten()
        )
    return th.stack(padded_edge_feats)
