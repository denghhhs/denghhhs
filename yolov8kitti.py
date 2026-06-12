import torch
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import pandas as pd
from tqdm import tqdm
import warnings
import yaml
import shutil
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class KITTIYOLOv8Research:
    def __init__(self, training_path, testing_path, output_dir='research_results'):
        self.training_path = Path(training_path)
        self.testing_path = Path(testing_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'detections').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'datasets').mkdir(exist_ok=True)

        self.kitti_classes = {
            'Pedestrian': 0, 'Cyclist': 1, 'Car': 2,
            'Motorcycle': 3, 'Bus': 5, 'Truck': 7
        }

        self.yolo_class_map = {
            'Pedestrian': 0, 'Cyclist': 1, 'Car': 2,
            'Van': 2, 'Truck': 3, 'Tram': 5, 'Bus': 5,
            'Motorcycle': 4, 'Bicycle': 1, 'Person_sitting': 0
        }

        self.driving_classes = {
            0: 'Pedestrian', 1: 'Cyclist', 2: 'Car',
            3: 'Truck', 4: 'Motorcycle', 5: 'Bus/Tram'
        }

        self.reset_metrics()

    def reset_metrics(self):
        self.research_metrics = {
            'detection_counts': defaultdict(int),
            'confidence_scores': defaultdict(list),
            'processing_times': [],
            'detection_sizes': [],
        }

    def convert_kitti_to_yolo(self):
        label_dir = self.training_path.parent / 'label_2'
        yolo_label_dir = self.output_dir / 'yolo_labels'
        yolo_label_dir.mkdir(exist_ok=True)

        print(f"[1/5] 转换标签: {label_dir} -> {yolo_label_dir}")

        converted_count = 0
        for label_file in tqdm(list(label_dir.glob('*.txt')), desc="  转换中"):
            img_file = self.training_path / f"{label_file.stem}.png"
            if not img_file.exists():
                continue

            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            yolo_annotations = []
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 15:
                        continue
                    obj_type = parts[0]
                    if obj_type == 'DontCare' or obj_type not in self.yolo_class_map:
                        continue

                    x1, y1 = float(parts[4]), float(parts[5])
                    x2, y2 = float(parts[6]), float(parts[7])
                    if x2 <= x1 or y2 <= y1:
                        continue

                    center_x = ((x1 + x2) / 2) / img_w
                    center_y = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    if width <= 0 or height <= 0:
                        continue

                    class_id = self.yolo_class_map[obj_type]
                    yolo_annotations.append(
                        f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    )

            output_file = yolo_label_dir / f"{label_file.stem}.txt"
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            converted_count += 1

        print(f"  完成: {converted_count} 个文件已转换")

    def train_model(self, epochs=50, batch_size=16, img_size=640):
        print(f"\n[2/5] 训练 YOLOv8n | epochs={epochs} batch={batch_size} imgsz={img_size}")

        dataset_dir = self.output_dir / 'datasets'
        train_img_dir = dataset_dir / 'images' / 'train'
        train_label_dir = dataset_dir / 'labels' / 'train'
        train_img_dir.mkdir(parents=True, exist_ok=True)
        train_label_dir.mkdir(parents=True, exist_ok=True)

        yolo_label_dir = self.output_dir / 'yolo_labels'

        for img_file in tqdm(list(self.training_path.glob('*.png')), desc="  复制图片"):
            shutil.copy2(img_file, train_img_dir / img_file.name)
        for label_file in tqdm(list(yolo_label_dir.glob('*.txt')), desc="  复制标签"):
            shutil.copy2(label_file, train_label_dir / label_file.name)

        print(f"  图片: {len(list(train_img_dir.glob('*.png')))} | 标签: {len(list(train_label_dir.glob('*.txt')))}")

        data_yaml = dataset_dir / 'kitti_data.yaml'
        data_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/train',
            'nc': 6,
            'names': ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Motorcycle', 'Bus/Tram']
        }
        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        model = YOLO('yolov8n.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  设备: {device}")

        model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=str(self.output_dir / 'models'),
            name='kitti_trained',
            exist_ok=True,
            patience=10,
            save=True,
            device=device
        )

        train_run_dir = self.output_dir / 'models' / 'kitti_trained'
        self.model = YOLO(str(train_run_dir / 'weights' / 'best.pt'))
        print(f"  训练完成, best.pt 已加载")

        results_csv = train_run_dir / 'results.csv'
        if results_csv.exists():
            self._plot_training_curves(results_csv)
        else:
            print("  警告: 未找到 results.csv, 跳过曲线绘制")

    def _plot_training_curves(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        epochs = df['epoch'].values if 'epoch' in df.columns else range(1, len(df) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('YOLOv8n — KITTI 训练曲线', fontsize=14, fontweight='bold')

        ax = axes[0]
        loss_colors = {
            'train/box_loss':  ('#1f77b4', 'train box'),
            'train/cls_loss':  ('#ff7f0e', 'train cls'),
            'train/dfl_loss':  ('#2ca02c', 'train dfl'),
            'val/box_loss':    ('#d62728', 'val box'),
            'val/cls_loss':    ('#9467bd', 'val cls'),
            'val/dfl_loss':    ('#8c564b', 'val dfl'),
        }
        for col, (color, label) in loss_colors.items():
            if col in df.columns:
                ax.plot(epochs, df[col], color=color, label=label, linewidth=1.2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss 曲线')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        map_colors = {
            'metrics/mAP50(B)':     ('#1f77b4', 'mAP@0.5'),
            'metrics/mAP50-95(B)':  ('#d62728', 'mAP@0.5:0.95'),
        }
        for col, (color, label) in map_colors.items():
            if col in df.columns:
                ax.plot(epochs, df[col], color=color, label=label, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('mAP 曲线')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  训练曲线已保存: {save_path}")

    def evaluate_model(self, dataset_type='training'):
        print(f"\n[3/5] 模型评估 ({dataset_type})")

        data_yaml = self.output_dir / 'datasets' / 'kitti_data.yaml'

        metrics = self.model.val(
            data=str(data_yaml),
            split='val',
            project=str(self.output_dir / 'metrics'),
            name=f'evaluation_{dataset_type}',
            exist_ok=True
        )

        print(f"  mAP@0.5: {metrics.box.map50:.4f}  |  mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}  |  Recall: {metrics.box.mr:.4f}")

        if hasattr(metrics.box, 'ap_class_index'):
            print("  各类别 AP@0.5:")
            for i, ap in enumerate(metrics.box.ap):
                class_name = self.driving_classes.get(i, f'Class_{i}')
                print(f"    {class_name:12s}: {ap:.4f}")

        return metrics

    def analyze_scene_complexity(self, detections, image_shape):
        complexity = {
            'total_objects': len(detections),
            'object_density': len(detections) / (image_shape[0] * image_shape[1]) * 10000,
            'class_diversity': len(set(d['class_id'] for d in detections)),
            'crowding_score': 0
        }
        if detections:
            for i, det1 in enumerate(detections):
                for det2 in detections[i + 1:]:
                    if self._calculate_iou(det1['bbox'], det2['bbox']) > 0.1:
                        complexity['crowding_score'] += 1
            complexity['crowding_score'] /= len(detections)
        return complexity

    def _calculate_iou(self, box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def process_dataset(self, dataset_type, num_samples=None):
        image_dir = self.training_path if dataset_type == 'training' else self.testing_path
        if not image_dir.exists():
            print(f"  错误: {image_dir} 不存在")
            return None, None

        image_files = sorted(image_dir.glob('*.png'))[:num_samples]
        if not image_files:
            print(f"  错误: {image_dir} 中无 PNG 文件")
            return None, None

        print(f"  处理 {len(image_files)} 张图片 ({dataset_type})")

        self.reset_metrics()
        scene_analyses = []
        all_detections = []

        for img_path in tqdm(image_files, desc=f"  推理中 ({dataset_type})"):
            start_time = time.time()
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model(image_rgb, conf=0.25, verbose=False)
            process_time = (time.time() - start_time) * 1000
            self.research_metrics['processing_times'].append(process_time)

            frame_detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    if cls_id in self.driving_classes:
                        detection = {
                            'image_id': img_path.stem,
                            'class_id': cls_id,
                            'class_name': self.driving_classes[cls_id],
                            'confidence': conf,
                            'bbox': bbox
                        }
                        frame_detections.append(detection)
                        self.research_metrics['detection_counts'][self.driving_classes[cls_id]] += 1
                        self.research_metrics['confidence_scores'][self.driving_classes[cls_id]].append(conf)

            scene_complexity = self.analyze_scene_complexity(frame_detections, image.shape)
            scene_complexity['image_id'] = img_path.stem
            scene_complexity['process_time_ms'] = process_time
            scene_analyses.append(scene_complexity)

            for det in frame_detections:
                x1, y1, x2, y2 = det['bbox']
                self.research_metrics['detection_sizes'].append((x2 - x1) * (y2 - y1))
            all_detections.extend(frame_detections)

            if len(scene_analyses) % 10 == 0:
                vis_image = results[0].plot()
                cv2.imwrite(
                    str(self.output_dir / 'visualizations' / f'{dataset_type}_{img_path.stem}.jpg'),
                    vis_image
                )

        return scene_analyses, all_detections

    def generate_report(self, scene_analyses, all_detections, dataset_type):
        if not scene_analyses:
            return None

        processing_times = self.research_metrics['processing_times']
        total_detections = sum(self.research_metrics['detection_counts'].values())
        avg_time = np.mean(processing_times)
        fps = 1000 / avg_time

        print(f"\n  ---- {dataset_type.upper()} 检测结果 ----")
        print(f"  图片: {len(scene_analyses)} | 检测数: {total_detections} | "
              f"平均耗时: {avg_time:.1f}ms ({fps:.1f} FPS)")
        print(f"  类别分布:")
        for class_name, count in sorted(self.research_metrics['detection_counts'].items(),
                                        key=lambda x: x[1], reverse=True):
            pct = (count / total_detections * 100) if total_detections > 0 else 0
            avg_conf = np.mean(self.research_metrics['confidence_scores'][class_name])
            print(f"    {class_name:12s}: {count:5d} ({pct:5.1f}%)  avg_conf={avg_conf:.3f}")

        self._create_analysis_charts(scene_analyses, dataset_type)

        report = {
            'dataset': f'KITTI {dataset_type}',
            'model': 'YOLOv8 Nano',
            'images_processed': len(scene_analyses),
            'total_detections': total_detections,
            'avg_time_ms': float(avg_time),
            'fps': float(fps),
            'class_stats': {
                cls: {
                    'count': count,
                    'avg_confidence': float(np.mean(self.research_metrics['confidence_scores'][cls]))
                }
                for cls, count in self.research_metrics['detection_counts'].items()
            }
        }
        with open(self.output_dir / 'metrics' / f'report_{dataset_type}.json', 'w') as f:
            json.dump(report, f, indent=2)

        if all_detections:
            df = pd.DataFrame(all_detections)
            df.to_csv(self.output_dir / 'detections' / f'detections_{dataset_type}.csv', index=False)

        return report

    def _create_analysis_charts(self, scene_analyses, dataset_type):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(f'YOLOv8n — KITTI {dataset_type.upper()} 检测分析', fontsize=13, fontweight='bold')

        ax = axes[0]
        times = self.research_metrics['processing_times']
        ax.hist(times, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(times), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {np.mean(times):.1f} ms')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Inference Time Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        classes = list(self.research_metrics['detection_counts'].keys())
        counts = list(self.research_metrics['detection_counts'].values())
        if classes:
            colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
            bars = ax.bar(classes, counts, color=colors, edgecolor='black')
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                        str(count), ha='center', fontsize=9)
        ax.set_ylabel('Count')
        ax.set_title('Detections per Class')
        ax.tick_params(axis='x', rotation=30)

        plt.tight_layout()
        save_path = self.output_dir / f'analysis_{dataset_type}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  分析图已保存: {save_path}")


if __name__ == "__main__":
    import os

    KITTI_BASE = "/home/hkust/denghhhs/dataset/KITTI"
    TRAINING_PATH = f"{KITTI_BASE}/training/image_2"
    TESTING_PATH = f"{KITTI_BASE}/testing/image_2"
    OUTPUT_DIR = "./research_results"

    if not os.path.exists(TRAINING_PATH):
        print(f"错误: 训练路径不存在 -> {TRAINING_PATH}")
        print("请修改 KITTI_BASE 为你的实际路径")
        exit(1)

    research = KITTIYOLOv8Research(TRAINING_PATH, TESTING_PATH, OUTPUT_DIR)

    print("=" * 55)
    print("  YOLOv8n × KITTI 完整分析流程")
    print("=" * 55)

    research.convert_kitti_to_yolo()

    research.train_model(epochs=50, batch_size=16, img_size=640)

    metrics = research.evaluate_model('training')

    print("\n[4/5] 训练集推理...")
    train_scenes, train_detections = research.process_dataset('training', num_samples=100)
    if train_scenes:
        research.generate_report(train_scenes, train_detections, 'training')

    print("\n[5/5] 测试集推理...")
    test_scenes, test_detections = research.process_dataset('testing', num_samples=100)
    if test_scenes:
        research.generate_report(test_scenes, test_detections, 'testing')

    print("\n" + "=" * 55)
    print(f"  全部完成! 结果目录: {OUTPUT_DIR}")
    print(f"    training_curves.png   — 损失 & mAP 曲线")
    print(f"    analysis_training.png — 训练集检测分析")
    print(f"    analysis_testing.png  — 测试集检测分析")
    print("=" * 55)
