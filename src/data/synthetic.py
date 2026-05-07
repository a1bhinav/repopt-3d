import torch


def make_synthetic_scenes(
    num_scenes: int = 5,
    points_per_scene: int = 500,
    d_feat: int = 64,
    num_classes: int = 5,
    seed: int = 0,
) -> list:
    """Generate scenes whose features are class-conditional Gaussians.

    Centers are spread out (std=5) and per-point noise is small (std=0.5),
    so a linear classifier can learn the mapping easily. This is a
    deliberate choice: the smoke test for the training loop relies on the
    data being learnable, otherwise a passing test would prove nothing
    about the loop wiring.

    Each scene is a dict with keys ``features`` (Tensor[N, d_feat]) and
    ``labels`` (LongTensor[N], values in ``[0, num_classes)``).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = torch.randn(num_classes, d_feat, generator=g) * 5.0
    scenes = []
    for _ in range(num_scenes):
        labels = torch.randint(0, num_classes, (points_per_scene,), generator=g)
        noise = torch.randn(points_per_scene, d_feat, generator=g) * 0.5
        features = centers[labels] + noise
        scenes.append({"features": features, "labels": labels.long()})
    return scenes
