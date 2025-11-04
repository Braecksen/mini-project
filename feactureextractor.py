import torch
import torch.nn as nn

class feactureExtractor(nn.Module):
   

    def __init__(self, layers_to_extract, backbone, device="cuda"):
        super(feactureExtractor, self).__init__()

        self.layers_to_extract = layers_to_extract
        self.backbone = backbone.to(device)
        self.device = device
        self.outputs = {}

        # Freeze backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Setup hook handles
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []

        # Remove old hooks
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.backbone.hook_handles = []

        # Register hooks for each layer
        for layer_name in layers_to_extract:
            layer = self.backbone._modules[layer_name]
            handle = layer.register_forward_hook(self._hook_creator(layer_name))
            self.backbone.hook_handles.append(handle)

    def _hook_creator(self, layer_name):
        def hook(module, input, output):
            self.outputs[layer_name] = output
        return hook

    def forward(self, x):
        self.outputs = {}
        x = x.to(self.device)
        with torch.no_grad():
            _ = self.backbone(x)
        return self.outputs
