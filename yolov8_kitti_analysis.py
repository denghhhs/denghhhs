import os
import sys
import cv2
import yaml
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image

KITTI_ROOT = Path(r"./dataset/KITTI")  


YOLO_DATASET_ROOT = Path(r"./kitti_yolo")


USE_EASY_CLASSES = True

# KITTI → 简化类别映射
KITTI_TO_SIMPLE = {
    "Car":           "Car",
    "Van":           "Car",
    "Truck":         "Car",
    "Tram":          "Car",
    "Pedestrian":    "Pedestrian",
    "Person_sitting":"Pedestrian",
    "Cyclist":       "Cyclist",
    "Misc":          None,       # 忽略
    "DontCare":      None,       # 忽略
}

# YOLOv8 模型选择: n/s/m/l/x (nano ~ xlarge)
MODEL_SIZE = "n"  # n=最快最小, x=最准最大

# 训练参数
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cuda"  # "cuda" 或 "cpu"


# ============================================================
# 1. KITTI 数据集探索
# ============================================================

def explore_kitti_dataset(kitti_root: Path):
    """探索 KITTI 数据集，打印统计信息"""
    print("=" * 60)
    print("📊 KITTI 数据集探索")
    print("=" * 60)

    train_img_dir = kitti_root / "training" / "image_2"
    train_label_dir = kitti_root / "training" / "label_2"
    test_img_dir = kitti_root / "testing" / "image_2"

    for name, path in [("训练图像", train_img_dir),
                       ("训练标注", train_label_dir),
                       ("测试图像", test_img_dir)]:
        if path.exists():
            count = len(list(path.glob("*.png"))) if "image" in str(path) else len(list(path.glob("*.txt")))
            print(f"  {name}: {path}  →  {count} 个文件")
        else:
            print(f"  ⚠ {name}: {path} 不存在")

    if not train_label_dir.exists():
        print("\n⚠ 标注目录不存在，跳过标注统计。")
        return

    # 统计类别分布
    class_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    truncated_stats = {"0": 0, "1": 0}
    occluded_stats = {"0": 0, "1": 0, "2": 0, "3": 0}

    label_files = sorted(train_label_dir.glob("*.txt"))
    total_boxes = 0
    dontcare_count = 0

    for label_file in label_files:
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                cls_name = parts[0]
                truncated = float(parts[1])
                occluded = int(parts[2])
                bbox = [float(x) for x in parts[4:8]]

                total_boxes += 1
                if cls_name == "DontCare":
                    dontcare_count += 1
                    continue

                class_counts[cls_name] += 1
                if truncated > 0:
                    truncated_stats["1"] += 1
                else:
                    truncated_stats["0"] += 1
                occluded_stats[str(occluded)] += 1

    print(f"\n📦 总标注框: {total_boxes}  (其中 DontCare: {dontcare_count})")
    print(f"\n📋 类别分布:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // max(1, max(class_counts.values()) // 30))
        print(f"  {cls:<18} {count:>6d}  {bar}")

    print(f"\n🔍 截断 (Truncated) 分布: 未截断={truncated_stats['0']}, 截断={truncated_stats['1']}")
    print(f"🔍 遮挡 (Occluded) 分布:  0={occluded_stats['0']}, 1={occluded_stats['1']}, 2={occluded_stats['2']}, 3={occluded_stats['3']}")

    return dict(class_counts)


def visualize_kitti_samples(kitti_root: Path, num_samples: int = 6):
    """可视化 KITTI 标注样本"""
    print("\n" + "=" * 60)
    print("🖼 可视化 KITTI 样本")
    print("=" * 60)

    train_img_dir = kitti_root / "training" / "image_2"
    train_label_dir = kitti_root / "training" / "label_2"

    if not train_img_dir.exists() or not train_label_dir.exists():
        print("⚠ 图像或标注目录不存在，跳过可视化。")
        return

    # KITTI 类别颜色 (BGR)
    COLOR_MAP = {
        "Car": (0, 255, 0),           # 绿色
        "Van": (0, 200, 0),
        "Truck": (0, 150, 0),
        "Tram": (0, 100, 0),
        "Pedestrian": (255, 0, 0),    # 蓝色
        "Person_sitting": (200, 0, 0),
        "Cyclist": (0, 0, 255),       # 红色
        "Misc": (255, 255, 0),
        "DontCare": (128, 128, 128),  # 灰色
    }

    img_files = sorted(train_img_dir.glob("*.png"))[:num_samples]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = train_label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls_name = parts[0]
                    if cls_name == "DontCare":
                        continue
                    x1, y1, x2, y2 = map(float, parts[4:8])
                    color = COLOR_MAP.get(cls_name, (255, 255, 255))
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(img, cls_name, (int(x1), int(y1) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        axes[idx].imshow(img)
        axes[idx].set_title(f"{img_path.stem}", fontsize=9)
        axes[idx].axis("off")

    plt.suptitle("KITTI 训练集标注样本", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = Path("./kitti_samples_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  可视化已保存至: {save_path}")


# ============================================================
# 2. KITTI → YOLO 格式转换
# ============================================================

def get_class_map():
    """生成类别名 → YOLO class_id 映射"""
    if USE_EASY_CLASSES:
        # 3 类: Car(0), Pedestrian(1), Cyclist(2)
        yolo_classes = ["Car", "Pedestrian", "Cyclist"]
    else:
        # 8 类: 去掉 DontCare 和 Misc
        yolo_classes = ["Car", "Van", "Truck", "Tram",
                       "Pedestrian", "Person_sitting",
                       "Cyclist", "Misc"]
    return {name: idx for idx, name in enumerate(yolo_classes)}, yolo_classes


def kitti_to_yolo_bbox(kitti_bbox, img_width, img_height):
    """
    KITTI bbox [x1, y1, x2, y2] (绝对像素坐标)
    → YOLO bbox [x_center, y_center, width, height] (归一化)
    """
    x1, y1, x2, y2 = kitti_bbox
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height


def convert_kitti_to_yolo(
    kitti_root: Path,
    yolo_root: Path,
    train_split: float = 0.85,
    seed: int = 42,
):
    """
    将 KITTI 格式转换为 YOLO 格式

    YOLO 目录结构:
      yolo_root/
        images/
          train/  ← 训练图像 (软链接或复制)
          val/    ← 验证图像
        labels/
          train/  ← YOLO 标注 (.txt)
          val/    ← YOLO 标注 (.txt)
        dataset.yaml ← 数据集描述文件
    """
    print("\n" + "=" * 60)
    print("🔄 KITTI → YOLO 格式转换")
    print("=" * 60)

    class_map, yolo_classes = get_class_map()
    print(f"  类别映射 ({len(yolo_classes)} 类): {yolo_classes}")
    print(f"  使用简化类别: {USE_EASY_CLASSES}")

    train_img_dir = kitti_root / "training" / "image_2"
    train_label_dir = kitti_root / "training" / "label_2"

    if not train_img_dir.exists():
        print(f"  ❌ 图像目录不存在: {train_img_dir}")
        return None
    if not train_label_dir.exists():
        print(f"  ❌ 标注目录不存在: {train_label_dir}")
        return None

    # 创建 YOLO 目录结构
    for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
        (yolo_root / subdir).mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件
    all_images = sorted(train_img_dir.glob("*.png"))
    print(f"  总图像数: {len(all_images)}")

    # 随机划分训练/验证集
    np.random.seed(seed)
    indices = np.random.permutation(len(all_images))
    split_point = int(len(all_images) * train_split)
    train_indices = set(indices[:split_point])
    val_indices = set(indices[split_point:])

    print(f"  训练集: {len(train_indices)} 张")
    print(f"  验证集: {len(val_indices)} 张")

    # 统计
    stats = {"train_boxes": 0, "val_boxes": 0, "skipped": 0, "dontcare": 0}

    for idx, img_path in enumerate(all_images):
        is_train = idx in train_indices
        subset = "train" if is_train else "val"

        # 读取图像尺寸
        img = Image.open(img_path)
        img_w, img_h = img.size

        # 处理标注
        label_path = train_label_dir / f"{img_path.stem}.txt"
        yolo_label_path = yolo_root / "labels" / subset / f"{img_path.stem}.txt"

        yolo_lines = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 15:
                        continue

                    kitti_cls = parts[0]
                    bbox = [float(x) for x in parts[4:8]]

                    # 类别映射
                    if USE_EASY_CLASSES:
                        mapped_cls = KITTI_TO_SIMPLE.get(kitti_cls)
                        if mapped_cls is None:
                            stats["dontcare"] += 1
                            continue
                    else:
                        mapped_cls = kitti_cls
                        if mapped_cls in ("DontCare",):
                            stats["dontcare"] += 1
                            continue

                    class_id = class_map.get(mapped_cls)
                    if class_id is None:
                        stats["skipped"] += 1
                        continue

                    # bbox 格式转换
                    yolo_bbox = kitti_to_yolo_bbox(bbox, img_w, img_h)

                    # 边界检查
                    if all(0 <= v <= 1 for v in yolo_bbox):
                        yolo_lines.append(f"{class_id} {' '.join(f'{v:.6f}' for v in yolo_bbox)}")
                        stats[f"{subset}_boxes"] += 1
                    else:
                        stats["skipped"] += 1

        # 写入 YOLO 标注
        with open(yolo_label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # 创建图像软链接 (Windows 上可能失败，使用复制作为备选)
        dest_img = yolo_root / "images" / subset / img_path.name
        if not dest_img.exists():
            try:
                os.symlink(img_path, dest_img)
            except OSError:
                shutil.copy2(img_path, dest_img)

        # 进度条
        if (idx + 1) % 500 == 0 or idx == len(all_images) - 1:
            print(f"  处理进度: {idx + 1}/{len(all_images)}")

    print(f"\n  ✅ 转换完成!")
    print(f"  训练框数: {stats['train_boxes']}")
    print(f"  验证框数: {stats['val_boxes']}")
    print(f"  跳过的无效框: {stats['skipped']}")
    print(f"  忽略的 DontCare: {stats['dontcare']}")

    # --- 生成 dataset.yaml ---
    dataset_yaml = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(yolo_classes),
        "names": yolo_classes,
    }

    yaml_path = yolo_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n  📄 数据集配置文件: {yaml_path}")
    print(f"  内容:\n{yaml.dump(dataset_yaml, default_flow_style=False, allow_unicode=True)}")

    return yaml_path


# ============================================================
# 3. YOLOv8 训练
# ============================================================

def train_yolov8(data_yaml: Path):
    """使用 YOLOv8 训练"""
    print("\n" + "=" * 60)
    print("🏋 YOLOv8 模型训练")
    print("=" * 60)

    from ultralytics import YOLO

    model_name = f"yolov8{MODEL_SIZE}.pt"
    print(f"  使用模型: {model_name}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  图像尺寸: {IMG_SIZE}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  设备: {DEVICE}")

    # 加载预训练模型
    model = YOLO(model_name)

    # 开始训练
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=4,
        optimizer="auto",          # AdamW/SGD 自动选择
        lr0=0.01,                  # 初始学习率
        lrf=0.01,                  # 最终学习率因子
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,               # 余弦退火
        close_mosaic=10,           # 最后 10 epoch 关闭 Mosaic 增强
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        project="kitti_yolov8_runs",
        name=f"train_{MODEL_SIZE}",
        exist_ok=True,
        verbose=True,
    )

    print(f"\n  ✅ 训练完成!")
    print(f"  最佳模型: {results.save_dir}/weights/best.pt")
    return results


# ============================================================
# 4. 模型评估与分析
# ============================================================

def evaluate_model(model_path: str, data_yaml: Path):
    """全面评估训练好的模型"""
    print("\n" + "=" * 60)
    print("📈 模型评估")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_path)

    # 在验证集上评估
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        save_json=True,
        save_hybrid=True,
    )

    print(f"\n  📊 总体性能:")
    print(f"  mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
    if metrics.box.mp is not None:
        print(f"  Precision:     {metrics.box.mp:.4f}")
    if metrics.box.mr is not None:
        print(f"  Recall:        {metrics.box.mr:.4f}")

    # 逐类别 AP
    if metrics.box.ap is not None:
        _, yolo_classes = get_class_map()
        print(f"\n  📋 逐类别 AP@0.5:")
        class_ap = metrics.box.ap50.tolist() if hasattr(metrics.box, 'ap50') else metrics.box.ap.tolist()
        for cls_name, ap in zip(yolo_classes, class_ap):
            bar = "█" * max(1, int(ap * 40))
            print(f"    {cls_name:<15} AP={ap:.4f}  {bar}")

    return metrics


def analyze_training_results(runs_dir: str = "kitti_yolov8_runs"):
    """分析训练过程中的损失曲线和指标变化"""
    print("\n" + "=" * 60)
    print("📉 训练过程分析")
    print("=" * 60)

    # 寻找最新的训练结果
    runs_path = Path(runs_dir) / f"train_{MODEL_SIZE}"
    results_csv = runs_path / "results.csv"

    if not results_csv.exists():
        # 尝试找最新的一次训练
        train_dirs = sorted(Path(runs_dir).glob("train_*"), key=os.path.getmtime, reverse=True)
        if train_dirs:
            results_csv = train_dirs[0] / "results.csv"

    if not results_csv.exists():
        print(f"  ⚠ 未找到训练结果 CSV: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    # 清理列名 (去除前后空格)
    df.columns = df.columns.str.strip()

    # 绘制训练曲线
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics_to_plot = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Classification Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
        ("metrics/precision(B)", "metrics/recall(B)", "Precision & Recall"),
        ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "mAP"),
        ("lr/pg0", None, "Learning Rate"),
    ]

    for ax, (col1, col2, title) in zip(axes.flatten(), metrics_to_plot):
        if col1 in df.columns:
            ax.plot(df.index, df[col1], label=col1.split("/")[-1], linewidth=1.5)
        if col2 and col2 in df.columns:
            ax.plot(df.index, df[col2], label=col2.split("/")[-1], linewidth=1.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"YOLOv8{MODEL_SIZE} 在 KITTI 上的训练曲线", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = Path("./kitti_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  训练曲线已保存至: {save_path}")

    # 打印最佳 epoch
    if "metrics/mAP50-95(B)" in df.columns:
        best_idx = df["metrics/mAP50-95(B)"].idxmax()
        best_map = df["metrics/mAP50-95(B)"].max()
        print(f"  最佳 mAP@0.5:0.95: {best_map:.4f}  (Epoch {best_idx})")
    if "metrics/mAP50(B)" in df.columns:
        best_idx50 = df["metrics/mAP50(B)"].idxmax()
        best_map50 = df["metrics/mAP50(B)"].max()
        print(f"  最佳 mAP@0.5:     {best_map50:.4f}  (Epoch {best_idx50})")


# ============================================================
# 5. 推理与可视化
# ============================================================

def run_inference_and_visualize(
    model_path: str,
    kitti_root: Path,
    num_samples: int = 6,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
):
    """在 KITTI 图像上运行推理并可视化结果"""
    print("\n" + "=" * 60)
    print("🔮 推理与可视化")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_path)

    # 从训练集/测试集取图像
    img_dir = kitti_root / "training" / "image_2"
    if not img_dir.exists():
        img_dir = kitti_root / "testing" / "image_2"

    if not img_dir.exists():
        print("  ❌ 未找到图像目录")
        return

    img_files = sorted(img_dir.glob("*.png"))[:num_samples]

    # 选取部分图像进行推理
    results = model.predict(
        source=[str(p) for p in img_files],
        imgsz=IMG_SIZE,
        conf=conf_threshold,
        iou=iou_threshold,
        device=DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        project="kitti_yolov8_runs",
        name=f"predict_{MODEL_SIZE}",
        exist_ok=True,
    )

    # 自定义可视化：并排显示原图和预测
    _, yolo_classes = get_class_map()
    # YOLOv8 默认颜色方案
    colors = [
        (0, 255, 0),     # 绿
        (255, 0, 0),     # 蓝
        (0, 0, 255),     # 红
        (255, 255, 0),   # 青
        (255, 0, 255),   # 品红
        (0, 255, 255),   # 黄
        (128, 0, 128),   # 紫
        (255, 128, 0),   # 橙
    ]

    fig, axes = plt.subplots(2, (num_samples + 1) // 2, figsize=(16, 8))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, (img_path, result) in enumerate(zip(img_files, results)):
        if idx >= len(axes):
            break

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 绘制预测框
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, clss, confs):
                x1, y1, x2, y2 = box.astype(int)
                color = colors[cls % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{yolo_classes[cls]} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        axes[idx].imshow(img)
        axes[idx].set_title(f"{img_path.stem}  ({len(boxes) if result.boxes else 0} objects)", fontsize=9)
        axes[idx].axis("off")

    # 隐藏多余的子图
    for idx in range(len(img_files), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"YOLOv8{MODEL_SIZE} KITTI 预测结果 (conf≥{conf_threshold})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = Path("./kitti_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  预测可视化已保存至: {save_path}")
    print(f"  详细结果保存在: kitti_yolov8_runs/predict_{MODEL_SIZE}/")


# ============================================================
# 6. 错误分析
# ============================================================

def error_analysis(model_path: str, data_yaml: Path, kitti_root: Path):
    """
    运行详细的错误分析:
      - 按类别/尺寸统计漏检 (False Negative) 和误检 (False Positive)
      - 分析模型在困难样本上的表现
    """
    print("\n" + "=" * 60)
    print("🔬 错误分析")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_path)
    _, yolo_classes = get_class_map()

    # 在验证集上运行预测
    results = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        save_json=True,
    )

    print(f"\n  📊 验证集总体: Precision={results.box.mp:.4f}, Recall={results.box.mr:.4f}")

    # 计算 F1-score
    if results.box.mp > 0 or results.box.mr > 0:
        f1 = 2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-8)
        print(f"  📊 F1-Score: {f1:.4f}")

    # 速度测试
    print(f"\n  ⚡ 推理速度测试:")
    speed = model.predict(
        source=str(kitti_root / "training" / "image_2"),
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False,
    )
    if speed and len(speed) > 0 and speed[0].speed:
        preprocess, inference, postprocess = speed[0].speed.values()
        total = preprocess + inference + postprocess
        print(f"    预处理: {preprocess:.2f} ms")
        print(f"    推理:   {inference:.2f} ms  (~{1000/inference:.0f} FPS)" if inference > 0 else "")
        print(f"    后处理: {postprocess:.2f} ms")
        print(f"    总耗时: {total:.2f} ms")

    return results


# ============================================================
# 7. 主流程
# ============================================================

def main():
    """主函数：串联整个分析流程"""
    print("""
╔══════════════════════════════════════════════════════════╗
║     YOLOv8 对 KITTI 数据集进行目标检测分析               ║
║     KITTI Object Detection Analysis with YOLOv8          ║
╚══════════════════════════════════════════════════════════╝
    """)

    # ---- 检查 KITTI 数据集 ----
    if not KITTI_ROOT.exists():
        print(f"\n❌ KITTI 数据集目录不存在: {KITTI_ROOT}")
        print("   请修改脚本开头的 KITTI_ROOT 变量为你的 KITTI 实际路径。")
        print("   KITTI 下载地址: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d")
        sys.exit(1)

    # ---- Step 1: 探索数据集 ----
    explore_kitti_dataset(KITTI_ROOT)
    visualize_kitti_samples(KITTI_ROOT, num_samples=6)

    # ---- Step 2: 转换为 YOLO 格式 ----
    data_yaml = convert_kitti_to_yolo(KITTI_ROOT, YOLO_DATASET_ROOT)

    if data_yaml is None:
        print("\n❌ 数据转换失败，退出。")
        sys.exit(1)

    # ---- Step 3: 训练模型 ----
    train_results = train_yolov8(data_yaml)

    # ---- 找到最佳模型路径 ----
    best_model = Path(train_results.save_dir) / "weights" / "best.pt"
    if not best_model.exists():
        # fallback: 尝试 last.pt
        best_model = Path(train_results.save_dir) / "weights" / "last.pt"

    print(f"\n  使用模型: {best_model}")

    # ---- Step 4: 评估模型 ----
    evaluate_model(str(best_model), data_yaml)

    # ---- Step 5: 分析训练过程 ----
    analyze_training_results()

    # ---- Step 6: 推理与可视化 ----
    run_inference_and_visualize(str(best_model), KITTI_ROOT)

    # ---- Step 7: 错误分析 ----
    error_analysis(str(best_model), data_yaml, KITTI_ROOT)

    print("\n" + "=" * 60)
    print("✅ 全部分析完成!")
    print("=" * 60)
    print(f"""
📁 输出文件:
  ├── {YOLO_DATASET_ROOT}/              ← YOLO 格式数据集
  ├── kitti_yolov8_runs/                ← 训练、评估、预测结果
  ├── kitti_samples_viz.png             ← 样本可视化
  ├── kitti_training_curves.png         ← 训练曲线
  └── kitti_predictions.png             ← 预测可视化
    """)


if __name__ == "__main__":
    main()
