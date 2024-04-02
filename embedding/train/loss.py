import torch
from info_nce import InfoNCE

# ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

def inbatch_negative_loss(q_embeddings, p_embeddings, temperature=0.02):
    # InfoNCE: refer to https://arxiv.org/pdf/1807.03748.pdf
    """
    # !Deprecated
    compare_scores = torch.mm(q_embeddings, p_embeddings.t())
    loss_qp = ce_loss(compare_scores / temperature, torch.arange(compare_scores.size(0)).to(device))
    loss_pq = ce_loss(compare_scores.t() / temperature, torch.arange(compare_scores.size(0)).to(device))
    return (loss_pq + loss_qp) / 2
    """
    loss = InfoNCE(temperature=temperature)
    return loss(q_embeddings, p_embeddings)

def hard_negative_loss(q_embeddings, p_embeddings, n_embeddings, temperature=0.02):
    """
    # !Deprecated
    inbatch_compare_scores = torch.mm(q_embeddings, p_embeddings.t())
    hard_compare_scores = torch.mm(q_embeddings, n_embeddings.t())
    total_compare_scores = torch.cat([inbatch_compare_scores, hard_compare_scores], dim=-1)

    loss_qp = ce_loss(total_compare_scores, torch.arange(inbatch_compare_scores.size(0)).to(device))
    loss_pq = ce_loss(inbatch_compare_scores.t(), torch.arange(inbatch_compare_scores.size(0)).to(device))

    return (loss_pq + loss_qp) / 2
    """
    loss = InfoNCE(negative_mode='paired', temperature=temperature)
    return loss(q_embeddings, p_embeddings, n_embeddings)

def log_sigmoid_loss(q_logits, p_logits, temperature = 1.):
    # refer to https://arxiv.org/pdf/2006.03632.pdf
    # q_logits refer to the logits of better representation, the higher the better
    return -torch.mean(torch.log(torch.sigmoid((q_logits - p_logits) / temperature)))

def gradcache_loss():
    pass
