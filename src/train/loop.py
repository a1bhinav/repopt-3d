import torch
import torch.nn as nn

from src.train.step import train_step


def train_loop(
    adapter: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    labeled_mask: torch.Tensor,
    num_epochs: int = 10,
    batch_size: int = 8192,
    trust_region=None,
    lambda_tr: float = 0.0,
    seed: int = 0,
    verbose: bool = False,
) -> list:
    """Train for ``num_epochs`` over the full feature pool, reshuffling
    every epoch with ``torch.Generator(seed + epoch)``. Each minibatch
    contains a mix of labeled and unlabeled points; ``train_step`` masks
    the supervised loss to labeled rows only.

    Per-epoch averages of ``loss``/``sup_loss``/``tr_loss`` are taken
    over batches (not over points), so an epoch with one zero-labeled
    batch will drag the supervised average down — that's intentional and
    matches what the optimizer actually saw.
    """
    n = features.shape[0]
    history = []
    for epoch in range(num_epochs):
        g = torch.Generator(device="cpu").manual_seed(seed + epoch)
        perm = torch.randperm(n, generator=g)
        feats_e = features[perm]
        lbls_e = labels[perm]
        lmask_e = labeled_mask[perm]

        sum_loss = 0.0
        sum_sup = 0.0
        sum_tr = 0.0
        total_labeled = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            result = train_step(
                adapter,
                classifier,
                optimizer,
                feats_e[i : i + batch_size],
                lbls_e[i : i + batch_size],
                lmask_e[i : i + batch_size],
                trust_region=trust_region,
                lambda_tr=lambda_tr,
            )
            sum_loss += result["loss"]
            sum_sup += result["sup_loss"]
            sum_tr += result["tr_loss"]
            total_labeled += result["n_labeled"]
            n_batches += 1

        record = {
            "epoch": epoch,
            "avg_loss": sum_loss / n_batches,
            "avg_sup_loss": sum_sup / n_batches,
            "avg_tr_loss": sum_tr / n_batches,
            "total_labeled_seen": total_labeled,
            "n_batches": n_batches,
        }
        history.append(record)
        if verbose:
            print(record)
    return history
