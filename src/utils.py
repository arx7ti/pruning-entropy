import torch
import torch.nn as nn


def get_layers(model: nn.Module):
    layers = list(model.layers())
    flat_layers = []

    if not layers:
        return model
    else:
        for layer in layers:
            try:
                flat_layers.extend(get_layers(layer))
            except TypeError:
                flat_layers.append(get_layers(layer))

    return flat_layers


@torch.no_grad()
def entropy(p):
    return -p * torch.log(p + 1e-12).sum()


@torch.no_grad()
def boltzmann_entropy(model, beta=10, mask=False, thresh=0.1):
    H = 0

    for layer in get_layers(model):
        if isinstance(layer, nn.Linear):
            magnitude = torch.norm(layer.weight.cpu(), dim=1)

            if mask:
                layer_mask = abs(mu) > 0.01
                n = layer_mask.sum() / len(layer_mask)
            else:
                n = 1

            # Boltzmann distribution
            p = torch.exp(beta * magnitude)
            p = p / p.sum()

            H += n * entropy(p)

    return H


@torch.no_grad()
def gaussian_entropy(model, mask=False, thresh=0.1):
    H = 0

    for layer in get_children(model):
        if isinstance(layer, nn.Linear):
            mu = layer.weight.mean(1).cpu()
            std = layer.weight.std(1).cpu()

            if mask:
                layer_mask = abs(mu) > 0.01
                n = layer_mask.sum() / len(layer_mask)
            else:
                n = 1

            # Gaussian distribution
            z = torch.norm(layer.weight) - mu / std
            z = z / torch.sqrt(torch.tensor(2.0))
            erf = torch.erf(z)
            p = 1 - 0.5 * (1 + erf)

            H += n * entropy(p)

    return H


def count_params(model):
    n_params = (p.numel() for p in model.parameters() if p.requires_grad)

    return sum(n_params)


def remove_dead_neurons(model, thresh=0.1):
    device = next(model.parameters()).device

    with torch.no_grad():
        new_layers = []
        prev_active_neurons = None
        i = 0

        for layer in get_layers(model):
            if isinstance(layer, nn.Linear):
                i += 1
                layer.to(device)

                # Identify active neurons based on weight magnitude
                active_neurons = torch.norm(layer.weight, dim=1) > thresh
                num_active = active_neurons.sum().item()

                print(
                    f"Layer {i} :: {num_active}/{layer.out_features} active neurons"
                )

                if num_active == 0:
                    print(f"Layer {i} completely pruned! Keeping one neuron.")
                    active_neurons[0] = True
                    num_active = 1  # Ensure at least one neuron remains

                # If previous layer was pruned, also prune input neurons
                if prev_active_neurons is not None:
                    new_in_features = prev_active_neurons.sum().item()
                    pruned_weight = layer.weight[
                        active_neurons, :][:, prev_active_neurons]
                else:
                    new_in_features = layer.in_features
                    pruned_weight = layer.weight[active_neurons, :]

                # Create pruned layer
                new_layer = nn.Linear(new_in_features,
                                      num_active,
                                      bias=(layer.bias is not None)).to(device)

                # Copy pruned weights
                new_layer.weight.data.copy_(pruned_weight.to(device))

                if layer.bias is not None:
                    new_layer.bias.data.copy_(
                        layer.bias[active_neurons].to(device))

                new_layers.append(new_layer)
                prev_active_neurons = active_neurons

            else:
                new_layers.append(layer.to(device))

        pruned_model = nn.Sequential(*new_layers).to(device)
        print("✅ Model Pruned!")

        return pruned_model
