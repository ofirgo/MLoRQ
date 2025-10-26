import torch


def sqnr_distance(in_model, data, output_ref, model_manager):
    output = model_manager.forward(in_model, data)
    out_ref = output_ref['logits'] if isinstance(output, dict) else output_ref
    out_ref = out_ref.to(model_manager.device)

    if isinstance(output, dict):
        delta = torch.mean(((output['logits'] - out_ref)/out_ref.max()) ** 2)
        norm = torch.mean((out_ref/out_ref.max()) ** 2)
    else:
        delta = torch.mean((output - out_ref) ** 2)
        norm = torch.mean(out_ref ** 2)

    return delta.item(), norm.item()
