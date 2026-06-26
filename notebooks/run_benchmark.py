import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms, models
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
    Subset
)

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = Path("./data/labeled_dataset_final")
OUTPUT_ROOT = Path("./runs/final_model_benchmark_09_06_2026")

NUM_CLASSES = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'a_bikes',
    'b_moto',
    'c_pass',
    'd_light_comm',
    'e_heavy_rigid',
    'f_articulated',
    'g_bus',
    'h_agri'
]

MODEL_BATCH_SIZES = {
    "mobilenet_v3_large": 128,
    "densenet121": 48,
    "resnet50": 48,
    "resnet101": 32,
    "regnet_y_8gf": 32,
    "efficientnet_v2_s": 32,
    "efficientnet_v2_m": 16,
    "convnext_tiny": 48,
    "convnext_small": 32,
    "convnext_base": 16,
    "swin_t": 32,
    "swin_s": 16,
}

BATCH_SIZE = 16 # Default batch size if not specified for a model

USE_AMP = torch.cuda.is_available()
torch.backends.cudnn.benchmark = torch.cuda.is_available()

print(f"Using device: {DEVICE}")

# =========================================================
# MODEL CONFIGS
# =========================================================
MODEL_CONFIGS = {

    # -----------------------------------------------------
    # ConvNeXt Family
    # -----------------------------------------------------

    "convnext_tiny": {
        "builder": lambda: models.convnext_tiny(weights='IMAGENET1K_V1'),
        "classifier": "convnext"
    },

    "convnext_small": {
        "builder": lambda: models.convnext_small(weights='IMAGENET1K_V1'),
        "classifier": "convnext"
    },

    "convnext_base": {
        "builder": lambda: models.convnext_base(weights='IMAGENET1K_V1'),
        "classifier": "convnext"
    },

    # -----------------------------------------------------
    # EfficientNetV2 Family
    # -----------------------------------------------------

    "efficientnet_v2_s": {
        "builder": lambda: models.efficientnet_v2_s(weights='IMAGENET1K_V1'),
        "classifier": "efficientnet"
    },

    "efficientnet_v2_m": {
        "builder": lambda: models.efficientnet_v2_m(weights='IMAGENET1K_V1'),
        "classifier": "efficientnet"
    },

    # -----------------------------------------------------
    # Swin Transformers
    # -----------------------------------------------------

    "swin_t": {
        "builder": lambda: models.swin_t(weights='IMAGENET1K_V1'),
        "classifier": "swin"
    },

    "swin_s": {
        "builder": lambda: models.swin_s(weights='IMAGENET1K_V1'),
        "classifier": "swin"
    },

    # -----------------------------------------------------
    # ResNet Family
    # -----------------------------------------------------

    "resnet50": {
        "builder": lambda: models.resnet50(weights='IMAGENET1K_V2'),
        "classifier": "resnet"
    },

    "resnet101": {
        "builder": lambda: models.resnet101(weights='IMAGENET1K_V2'),
        "classifier": "resnet"
    },

    # -----------------------------------------------------
    # RegNet
    # -----------------------------------------------------

    "regnet_y_8gf": {
        "builder": lambda: models.regnet_y_8gf(weights='IMAGENET1K_V2'),
        "classifier": "regnet"
    },

    # -----------------------------------------------------
    # DenseNet
    # -----------------------------------------------------

    "densenet121": {
        "builder": lambda: models.densenet121(weights='IMAGENET1K_V1'),
        "classifier": "densenet"
    },

    # -----------------------------------------------------
    # MobileNet (baseline efficiency comparison)
    # -----------------------------------------------------

    "mobilenet_v3_large": {
        "builder": lambda: models.mobilenet_v3_large(weights='IMAGENET1K_V2'),
        "classifier": "mobilenet"
    }
}

# =========================================================
# TRANSFORMS
# =========================================================

def get_transforms():

    norm = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    train_tf = transforms.Compose([
        transforms.Resize((236, 236)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),

        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=(3, 3),
                sigma=(0.1, 2.0)
            )
        ], p=0.1),

        transforms.ToTensor(),
        norm
    ])

    val_tf = transforms.Compose([
        transforms.Resize((236, 236)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm
    ])

    return train_tf, val_tf

# =========================================================
# DATASET
# =========================================================


def create_datasets():

    train_tf, val_tf = get_transforms()

    full_dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=train_tf
    )

    assert len(full_dataset.classes) == NUM_CLASSES

    print("Detected classes:")
    print(full_dataset.classes)

    assert full_dataset.classes == CLASS_NAMES, (
        "Dataset folder order does not match CLASS_NAMES. "
        f"Found: {full_dataset.classes}"
    )

    targets = np.array(full_dataset.targets)
    indices = np.arange(len(full_dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=targets
        )

    train_dataset = Subset(
        full_dataset,
        train_indices
    )

    val_base = datasets.ImageFolder(
        DATA_DIR,
        transform=val_tf
    )

    val_dataset = Subset(
        val_base,
        val_indices
    )

    # Save split indices for reproducibility
    split_dir = OUTPUT_ROOT / "_splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.DataFrame({
        "path": [full_dataset.samples[i][0] for i in indices],
        "label_index": [full_dataset.samples[i][1] for i in indices],
        "label_name": [full_dataset.classes[full_dataset.samples[i][1]] for i in indices],
        "split": [
            "train" if i in set(train_indices) else "val"
            for i in indices
        ]
    })

    split_df.to_csv(
        split_dir / "split.csv",
        index=False
    )

    return full_dataset, train_dataset, val_dataset

# =========================================================
# SAMPLER
# =========================================================

def create_sampler(full_dataset, train_dataset):

    train_targets = [
        full_dataset.targets[i]
        for i in train_dataset.indices
    ]

    class_counts = np.bincount(train_targets)

    class_weights = 1.0 / class_counts

    sample_weights = [
        class_weights[t]
        for t in train_targets
    ]

    sampler = WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    return sampler

# =========================================================
# MODEL SETUP
# =========================================================

def build_model(model_name):

    cfg = MODEL_CONFIGS[model_name]

    model = cfg["builder"]()

    cls_type = cfg["classifier"]

    # -----------------------------------------------------
    # ConvNeXt
    # -----------------------------------------------------

    if cls_type == "convnext":

        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features,
            NUM_CLASSES
        )

    # -----------------------------------------------------
    # EfficientNet
    # -----------------------------------------------------

    elif cls_type == "efficientnet":

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES
        )

    # -----------------------------------------------------
    # Swin Transformer
    # -----------------------------------------------------

    elif cls_type == "swin":

        model.head = nn.Linear(
            model.head.in_features,
            NUM_CLASSES
        )

    # -----------------------------------------------------
    # ResNet
    # -----------------------------------------------------

    elif cls_type == "resnet":

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    # -----------------------------------------------------
    # RegNet
    # -----------------------------------------------------

    elif cls_type == "regnet":

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    # -----------------------------------------------------
    # DenseNet
    # -----------------------------------------------------

    elif cls_type == "densenet":

        model.classifier = nn.Linear(
            model.classifier.in_features,
            NUM_CLASSES
        )

    # -----------------------------------------------------
    # MobileNet
    # -----------------------------------------------------

    elif cls_type == "mobilenet":

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            NUM_CLASSES
        )

    return model.to(DEVICE)

def train_single_model(model_name, full_dataset, train_dataset, val_dataset):

    print("\n================================================")
    print(f"TRAINING: {model_name}")
    print("================================================")

    output_dir = OUTPUT_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.csv"

    if summary_path.exists() and (output_dir / "best_model.pth").exists():
        print(f"Skipping {model_name}: already completed.")
        return pd.read_csv(summary_path).iloc[0].to_dict()

    batch_size = MODEL_BATCH_SIZES.get(model_name, BATCH_SIZE)
    print(f"Batch size: {batch_size}")

    sampler = create_sampler(full_dataset, train_dataset)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(model_name)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.05
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    best_f1 = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    history = []
    start = time.time()

    for epoch in range(NUM_EPOCHS):

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):

                    with torch.autocast(
                        device_type="cuda",
                        enabled=USE_AMP
                    ):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)

                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = np.mean(
                np.array(all_preds) == np.array(all_labels)
            )

            epoch_f1 = f1_score(
                all_labels,
                all_preds,
                average="macro",
                zero_division=0
            )

            print(
                f"{phase} "
                f"Loss: {epoch_loss:.4f} | "
                f"Acc: {epoch_acc:.4f} | "
                f"Macro-F1: {epoch_f1:.4f}"
            )

            history.append({
                "epoch": epoch + 1,
                "phase": phase,
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "f1_macro": epoch_f1
            })

            if phase == "val" and epoch_f1 > best_f1:

                best_f1 = epoch_f1
                best_weights = copy.deepcopy(model.state_dict())

                torch.save(
                    model.state_dict(),
                    output_dir / "best_model.pth"
                )

                print(">>> NEW BEST MODEL")

    elapsed = time.time() - start

    print(
        f"\nFinished in "
        f"{elapsed // 60:.0f}m "
        f"{elapsed % 60:.0f}s"
    )

    print(f"Best Macro-F1: {best_f1:.4f}")

    model.load_state_dict(best_weights)

    # =====================================================
    # SAVE HISTORY
    # =====================================================

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        output_dir / "training_log.csv",
        index=False
    )

    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():

        for inputs, labels in val_loader:

            inputs = inputs.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                enabled=USE_AMP
            ):
                outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    final_accuracy = np.mean(
        np.array(y_true) == np.array(y_pred)
    )

    final_macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    final_weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} | Macro-F1={final_macro_f1:.4f}")
    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png"
    )

    plt.close()

    # =====================================================
    # REPORT
    # =====================================================

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    with open(
        output_dir / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    # =====================================================
    # SUMMARY
    # =====================================================

    if torch.cuda.is_available():
        peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    else:
        peak_gpu_memory_gb = 0.0

    summary = {
        "model": model_name,
        "status": "OK",
        "best_macro_f1": best_f1,
        "final_macro_f1": final_macro_f1,
        "final_weighted_f1": final_weighted_f1,
        "accuracy": final_accuracy,
        "dataset_size": len(full_dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "epochs": NUM_EPOCHS,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_gb": peak_gpu_memory_gb
    }

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(
        output_dir / "summary.csv",
        index=False
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    full_dataset, train_dataset, val_dataset = create_datasets()
    print(f"Dataset size: {len(full_dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    all_results = []

    for model_name in MODEL_CONFIGS.keys():

        try:
            result = train_single_model(
                model_name,
                full_dataset,
                train_dataset,
                val_dataset
            )

            all_results.append(result)

        except torch.cuda.OutOfMemoryError as e:
            print(f"\nOOM while training {model_name}: {e}")

            torch.cuda.empty_cache()

            all_results.append({
                "model": model_name,
                "status": "OOM",
                "best_macro_f1": np.nan,
                "final_macro_f1": np.nan,
                "final_weighted_f1": np.nan,
                "accuracy": np.nan,
                "dataset_size": len(full_dataset),
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "epochs": NUM_EPOCHS,
                "batch_size": MODEL_BATCH_SIZES.get(model_name, BATCH_SIZE),
                "elapsed_seconds": np.nan,
                "peak_gpu_memory_gb": np.nan
            })

        except Exception as e:
            print(f"\nFAILED while training {model_name}: {type(e).__name__}: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            all_results.append({
                "model": model_name,
                "status": f"FAILED: {type(e).__name__}",
                "best_macro_f1": np.nan,
                "final_macro_f1": np.nan,
                "final_weighted_f1": np.nan,
                "accuracy": np.nan,
                "dataset_size": len(full_dataset),
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "epochs": NUM_EPOCHS,
                "batch_size": MODEL_BATCH_SIZES.get(model_name, BATCH_SIZE),
                "elapsed_seconds": np.nan,
                "peak_gpu_memory_gb": np.nan
            })

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results_df = pd.DataFrame(all_results)

        results_df.to_csv(
            OUTPUT_ROOT / "final_model_comparison_partial.csv",
            index=False
        )

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="best_macro_f1",
        ascending=False,
        na_position="last"
    )

    results_df.to_csv(
        OUTPUT_ROOT / "final_model_comparison.csv",
        index=False
    )

    print("\n================================================")
    print("FINAL RESULTS")
    print("================================================")

    print(results_df)