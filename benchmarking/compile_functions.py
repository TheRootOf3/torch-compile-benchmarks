import torch


def compile_model_fn(model):
    return torch.compile(model, mode="reduce-overhead")


def compile_layers_fn(model):
    for layer in model.model.layers:
        layer.forward = torch.compile(layer.forward, mode="reduce-overhead")

    return model


def compile_mlp_attn_fn(model):
    for layer in model.model.layers:
        layer.mlp.forward = torch.compile(layer.mlp.forward, mode="reduce-overhead")
        layer.self_attn.forward = torch.compile(
            layer.self_attn.forward, mode="reduce-overhead"
        )

    return model
