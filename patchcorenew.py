from feactureextractor import feactureExtractor
from rich.progress import track
import numpy as np
import torch
from preprocessing import preprocessing
from typing import Union
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from knnsearcher import KCenterGreedy

class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)

class IdentitySampler:
    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        return features

class PatchMaker:
    def __init__(self, patchsize, stride):
        self.patchsize = patchsize
        self.stride = stride

    def make_patches(self, features, return_spatial_info=False):
        padding = (self.patchsize - 1) // 2
        unfolder = torch.nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding)
        unfolded_features = unfolder(features)

        n_patches = []
        for s in features.shape[-2:]:
            n_p = (s + 2 * padding - (self.patchsize - 1) - 1) / self.stride + 1
            n_patches.append(int(n_p))

        unfolded_features = unfolded_features.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, n_patches
        return unfolded_features

class patchCore:
    def __init__(self, layers, backbone, embed_dim, target_dim, patchsize, patchstride, sampling_ratio, shape, device="cuda"):
        self.layers_to_extract = layers
        self.feature_extractor = feactureExtractor(layers, backbone, device)
        self.device = device
        self.input_shape = shape
        self.memory_bank = None
        self.patch_maker = PatchMaker(patchsize, patchstride)
        self.embed_dim = embed_dim
        self.feature_dimensions = [512, 1024]  # adjust according to your layers
        self.preprocess = preprocessing(input_dims=self.feature_dimensions, output_dim=embed_dim).to(self.device)
        self.sampler = IdentitySampler()
        self.aggregator = Aggregator(target_dim=target_dim)
        self.target_dim = target_dim
        self.threshold = None

    def fit(self, data_loader):
        """Build memory bank from training images."""
        self.feature_extractor.eval()
        all_features = []

        def extract_features(img):
            with torch.no_grad():
                img = img.to(torch.float).to(self.device)
                batch_features = self.feature_extractor.forward(img)
                batch_features = [batch_features[layer] for layer in self.layers_to_extract]

                patched_features = [self.patch_maker.make_patches(x, return_spatial_info=True) for x in batch_features]
                patch_shapes = [x[1] for x in patched_features]
                patched_features = [x[0] for x in patched_features]
                reference = patch_shapes[0]

                
                for i in range(1, len(patched_features)):
                    feat = patched_features[i]
                    ps = patch_shapes[i]
                    feat = feat.reshape(feat.shape[0], ps[0], ps[1], *feat.shape[2:])
                    feat = feat.permute(0, -3, -2, -1, 1, 2)
                    base_shape = feat.shape
                    feat = feat.reshape(-1, *feat.shape[-2:])
                    feat = F.interpolate(feat.unsqueeze(1), size=(reference[0], reference[1]), mode="bilinear", align_corners=False)
                    feat = feat.squeeze(1)
                    feat = feat.reshape(*base_shape[:-2], reference[0], reference[1])
                    feat = feat.permute(0, -2, -1, 1, 2, 3)
                    feat = feat.reshape(len(feat), -1, *feat.shape[-3:])
                    patched_features[i] = feat

                features = [x.reshape(-1, *x.shape[-3:]) for x in patched_features]
                features = self.preprocess(features)
                features = self.aggregator(features)
                return features

        for batch in track(data_loader, description="Computing features..."):
            images = batch["image"].to(self.device, dtype=torch.float)
            batch_feats = extract_features(images)
            all_features.append(batch_feats.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)

        n_select = int(all_features.shape[0] * 0.1)
        selector = KCenterGreedy(X=all_features, y=None, seed=0, metric="euclidean")
        selected_indices = selector.select_batch(model=None, already_selected=[], N=n_select)

        all_features = all_features[selected_indices]
        print(f"Reduced Memory bank shape using k-Center-Greedy: {all_features.shape}")

        self.memory_bank = all_features


    def compute_distance_scores(self, folder_or_list, transform, backbone):

        
        memory_bank = torch.tensor(self.memory_bank, device=self.device, dtype=torch.float32)

        y_scores = []

        
        if isinstance(folder_or_list, (list, tuple)):
            all_files = folder_or_list
        else:
            all_files = list(folder_or_list.iterdir())

        for pth in track(all_files, description="Computing scores...", total=len(all_files)):
            img = transform(Image.open(pth)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                raw_feats = self.feature_extractor.forward(img)
                raw_feats = [raw_feats[layer] for layer in self.layers_to_extract]
                patched = [self.patch_maker.make_patches(x, return_spatial_info=False) for x in raw_feats]
                feats = self.preprocess(patched)
                feats = self.aggregator(feats)

            distances = torch.cdist(feats, memory_bank, p=2.0)
            dist_score, _ = torch.min(distances, dim=1)
            s_star = torch.max(dist_score)
            y_scores.append(s_star.item())

        return y_scores


    

