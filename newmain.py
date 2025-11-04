import torch
from torch.utils.data import DataLoader
from dataset import MVTecDataset, DatasetSplit
from torchvision import models, transforms
from patchcorenew import patchCore
from rich.progress import track
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

        

class TestPatchCore:

    def __init__(self):
        # transform for images
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        # dataset path
        self.dataset_root = "data/mvtec_anomaly_detection"
        self.classname = "hazelnut"

        # Load datasets
        self.train_dataset = MVTecDataset(
            source=self.dataset_root,
            classname=self.classname,
            split=DatasetSplit.TRAIN
        )
        self.test_dataset = MVTecDataset(
            source=self.dataset_root,
            classname=self.classname,
            split=DatasetSplit.TEST
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        print(f"Train size: {len(self.train_dataset)}, Test size: {len(self.test_dataset)}")

        # PatchCore parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = models.wide_resnet50_2(pretrained=True)
        layers_to_extract = ["layer2", "layer3"]
        shape = (3,224,224)
        embed_dim = 1024
        target_dim = 1024
        patchsize = 3
        patchstride = 1
        sampling_ratio = 0.1

        print("Patch Core parameters loaded")

        # Init PatchCore
        self.pc = patchCore(layers_to_extract, self.backbone, embed_dim, target_dim,
                            patchsize, patchstride, sampling_ratio, shape=shape, device=self.device)

        print("Patch Core instance initialized")

    def run(self):
        # 1. Train / Build memory bank
        self.pc.fit(self.train_loader)

        # 2. Evaluate on test set
        base_path = Path(r"C:\Users\braec\Desktop\totalrecall\data\mvtec_anomaly_detection\hazelnut\test")
        classes = ['crack', 'cut', 'good', 'hole', 'print']

        y_true = []
        paths = []

        for cls in classes:
            folder_path_test = base_path / cls   

            for pth in folder_path_test.iterdir():
                paths.append(pth)
                label = 0 if cls == "good" else 1
                y_true.append(label)

        # 3. Compute anomaly scores for all test images
        y_scores = self.pc.compute_distance_scores(paths, self.transform, self.backbone)


        auc = roc_auc_score(y_true, y_scores)

        threshold = np.percentile(y_scores, 95)
        y_pred = [1 if s > threshold else 0 for s in y_scores]
        f1 = f1_score(y_true, y_pred)

        print("AUC:", auc)
        print("F1:", f1)
        print("Sample scores:", y_scores[:5])

    

        RocCurveDisplay.from_predictions(y_true, y_scores)
        plt.title("ROC Curve - Hazelnut")
        plt.grid(True)
        plt.show()

        
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:\n", cm)

        
        
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred Good", "Pred Defect"],
                    yticklabels=["True Good", "True Defect"])
        plt.title("Confusion Matrix")
        plt.show()
        return auc, f1, y_scores
        


# Run it
if __name__ == "__main__":
    test = TestPatchCore()
    test.run()
