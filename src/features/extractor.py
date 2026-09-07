import torch
import numpy as np
import os
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.features = []
        modules = dict([*self.model.named_modules()])
        if self.layer_name not in modules:
            raise KeyError(f"Layer '{self.layer_name}' not found in model. Available layers: {list(modules.keys())}")
        self._layer = modules[self.layer_name]

    def extract(self, dataloader, device):
        self.model.eval()
        self.features = []
        all_labels = []
        all_preds = []

        def hook(module, input, output):
            self.features.append(output.detach().cpu().numpy())

        # Register hook per extraction call (cleanly removed in finally block)
        hook_handle = self._layer.register_forward_hook(hook)

        try:
            with torch.no_grad():
                for inputs, labels in tqdm(dataloader, desc=f"Extracting [{self.layer_name}]", leave=False):
                    inputs = inputs.to(device)
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_labels.extend(labels.numpy())
                    all_preds.extend(preds)
        finally:
            hook_handle.remove()

        if len(self.features) == 0:
            raise RuntimeError(f"No features were captured from layer '{self.layer_name}'. Ensure dataloader is not empty.")

        features_np = np.concatenate(self.features, axis=0)
        labels_np = np.array(all_labels)
        preds_np = np.array(all_preds)

        return features_np, labels_np, preds_np

def save_embeddings(features, labels, preds, split_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{split_name}_features.npy"), features)
    np.save(os.path.join(output_dir, f"{split_name}_labels.npy"), labels)
    np.save(os.path.join(output_dir, f"{split_name}_preds.npy"), preds)

def load_embeddings(split_name, input_dir):
    features = np.load(os.path.join(input_dir, f"{split_name}_features.npy"))
    labels = np.load(os.path.join(input_dir, f"{split_name}_labels.npy"))
    preds = np.load(os.path.join(input_dir, f"{split_name}_preds.npy"))
    return features, labels, preds
