import torch
from tqdm import tqdm
from compression.module.compression_wrapper import CompressionWrapper
import constants as C

from compression.weighted_svd.hessian_config import HessianConfig


def weight_lfh(in_images, in_model, hconfig: HessianConfig = HessianConfig()):
    print("Start Weight Label Free Hessian computation...")
    ##############################
    # Compute
    ##############################
    target_layers = {n: m for n, m in in_model.named_modules() if isinstance(m, CompressionWrapper)
                     and m.enable_low_rank}

    results_dict = {n: [] for n, m in target_layers.items()}

    for image_index in range(in_images.shape[0]):
        print(f"\nImage {image_index + 1} / {in_images.shape[0]}")
        image_results_dict = {n: 0 for n, m in target_layers.items()}
        _x = in_images[image_index, :].unsqueeze(dim=0)
        y_hat = in_model(_x.to(C.DEVICE))

        error_dict = {n: float('inf') for n in target_layers.keys()}

        for i in tqdm(range(hconfig.n_iter)):
            v = torch.randn_like(y_hat)
            out_v = torch.mean(y_hat.unsqueeze(dim=1) @ v.unsqueeze(dim=-1))

            out_v.backward(retain_graph=True)

            for n, m in target_layers.items():
                jac_v = m.base_module.weight.grad
                lfh = (jac_v ** 2.0).detach()  # per-element

                new_value = (i * image_results_dict[n] + lfh) / (i + 1)
                if i > hconfig.min_iterations:
                    error_dict[n] = torch.mean(torch.abs(new_value - image_results_dict[n])) / (
                                torch.mean(torch.abs(new_value)) + 1e-6)

                image_results_dict[n] = new_value

            if i > hconfig.min_iterations and torch.max(torch.tensor(list(error_dict.values()))) < hconfig.tol:
                break

        for k, v in image_results_dict.items():
            results_dict[k].append(v)

    return {k: torch.mean(torch.stack(v, dim=0), dim=0) for k, v in results_dict.items()}


def set_model_hessian_scores(model, in_images, n_iter=1e-4):
    hc = HessianConfig(n_iter=n_iter)
    results = weight_lfh(in_images, model, hc)
    for n, m in tqdm(model.named_modules()):
        if isinstance(m, CompressionWrapper) and m.enable_low_rank:
            m.add_weights_hessian_information(results[n])
