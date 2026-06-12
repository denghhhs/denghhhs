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
            'Pedestrian': 0,
            'Cyclist': 1,
            'Car': 2,
            'Motorcycle': 3,
            'Bus': 5,
            'Truck': 7
        }
        
        self.yolo_class_map = {
            'Pedestrian': 0,
            'Cyclist': 1,
            'Car': 2,
            'Van': 2,
            'Truck': 3,
            'Tram': 5,
            'Bus': 5,
            'Motorcycle': 4,
            'Bicycle': 1,
            'Person_sitting': 0
        }
        
        self.driving_classes = {
            0: 'Pedestrian',
            1: 'Cyclist', 
            2: 'Car',
            3: 'Truck',
            4: 'Motorcycle',
            5: 'Bus/Tram'
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
    
        print("Converting KITTI labels to YOLO format...")
        print(f"Source labels: {label_dir}")
        print(f"Output labels: {yolo_label_dir}")
    
        converted_count = 0
        for label_file in tqdm(list(label_dir.glob('*.txt'))):
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
                    if obj_type == 'DontCare':
                        continue
                    if obj_type not in self.yolo_class_map:
                        continue
                
                    x1 = float(parts[4])
                    y1 = float(parts[5])
                    x2 = float(parts[6])
                    y2 = float(parts[7])
                
                    if x2 <= x1 or y2 <= y1:
                        continue
                
                    center_x = ((x1 + x2) / 2) / img_w
                    center_y = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                
                    if width <= 0 or height <= 0 or center_x < 0 or center_x > 1 or center_y < 0 or center_y > 1:
                        continue
                
                    class_id = self.yolo_class_map[obj_type]
                    yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
            output_file = yolo_label_dir / f"{label_file.stem}.txt"
            if yolo_annotations:
                with open(output_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
            else:
                with open(output_file, 'w') as f:
                    f.write('')
        
            converted_count += 1
    
        print(f"Label conversion complete! Converted {converted_count} files.")
        print(f"Sample check - first label file content:")
        sample_files = list(yolo_label_dir.glob('*.txt'))
        if sample_files:
            with open(sample_files[0], 'r') as f:
                print(f"  {sample_files[0].name}: {f.read()[:200]}")

    def train_model(self, epochs=50, batch_size=16, img_size=640):
        print("\nStarting YOLOv8 Nano training on KITTI...")
    
        dataset_dir = self.output_dir / 'datasets'
        train_img_dir = dataset_dir / 'images' / 'train'
        train_label_dir = dataset_dir / 'labels' / 'train'
        train_img_dir.mkdir(parents=True, exist_ok=True)
        train_label_dir.mkdir(parents=True, exist_ok=True)
    
        yolo_label_dir = self.output_dir / 'yolo_labels'
    
        print("Preparing training images...")
        img_count = 0
        for img_file in tqdm(list(self.training_path.glob('*.png'))):
            shutil.copy2(img_file, train_img_dir / img_file.name)
            img_count += 1
        print(f"Copied {img_count} images")
    
        print("Preparing training labels...")
        label_count = 0
        for label_file in tqdm(list(yolo_label_dir.glob('*.txt'))):
            shutil.copy2(label_file, train_label_dir / label_file.name)
            label_count += 1
        print(f"Copied {label_count} labels")
    
        print(f"Verification - train images: {len(list(train_img_dir.glob('*.png')))}")
        print(f"Verification - train labels: {len(list(train_label_dir.glob('*.txt')))}")
    
        sample_labels = list(train_label_dir.glob('*.txt'))
        if sample_labels:
            with open(sample_labels[0], 'r') as f:
                content = f.read()
                print(f"Sample label content ({sample_labels[0].name}): {content[:200]}")
    
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
    
        print(f"\nData YAML content:")
        with open(data_yaml, 'r') as f:
            print(f.read())
    
        model = YOLO(r'D:\yolo\yolov8n.pt')
    
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=str(self.output_dir / 'models'),
            name='kitti_trained',
            exist_ok=True,
            patience=10,
            save=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
        self.model = YOLO(str(self.output_dir / 'models' / 'kitti_trained' / 'weights' / 'best.pt'))
        print("Training complete!")
    
        return results
    
    def evaluate_model(self, dataset_type='training'):
        print(f"\nEvaluating model on {dataset_type} dataset...")
        
        data_yaml = self.output_dir / 'datasets' / 'kitti_data.yaml'
        
        metrics = self.model.val(
            data=str(data_yaml),
            split='val',
            project=str(self.output_dir / 'metrics'),
            name=f'evaluation_{dataset_type}',
            exist_ok=True
        )
        
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS - {dataset_type.upper()}")
        print("="*60)
        print(f"mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"Precision:    {metrics.box.mp:.4f}")
        print(f"Recall:       {metrics.box.mr:.4f}")
        
        if hasattr(metrics.box, 'ap_class_index'):
            print("\nPer-class AP@0.5:")
            for i, ap in enumerate(metrics.box.ap):
                class_name = self.driving_classes.get(i, f'Class_{i}')
                print(f"  {class_name:12s}: {ap:.4f}")
        
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
                for j, det2 in enumerate(detections[i+1:], i+1):
                    if self._calculate_iou(det1['bbox'], det2['bbox']) > 0.1:
                        complexity['crowding_score'] += 1
            
            complexity['crowding_score'] /= len(detections)
        
        return complexity
    
    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def process_dataset(self, dataset_type, num_samples=None):
        if dataset_type == 'training':
            image_dir = self.training_path
        else:
            image_dir = self.testing_path
        
        if not image_dir.exists():
            print(f"Error: {image_dir} not found")
            return None, None
        
        image_files = sorted(image_dir.glob('*.png'))[:num_samples]
        
        if not image_files:
            print(f"Error: No PNG files in {image_dir}")
            return None, None
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        self.reset_metrics()
        scene_analyses = []
        all_detections = []
        
        for img_path in tqdm(image_files, desc=f"Processing {dataset_type}"):
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
                self.research_metrics['detection_sizes'].append((x2-x1) * (y2-y1))
            
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
        
        print(f"\n{'='*60}")
        print(f"KITTI {dataset_type.upper()} - Detection Results")
        print(f"{'='*60}")
        print(f"Images processed: {len(scene_analyses)}")
        print(f"Total detections: {total_detections}")
        print(f"Average time: {avg_time:.2f} ms ({fps:.1f} FPS)")
        print(f"\nClass Distribution:")
        
        for class_name, count in sorted(self.research_metrics['detection_counts'].items(), 
                                       key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            avg_conf = np.mean(self.research_metrics['confidence_scores'][class_name])
            print(f"  {class_name:12s}: {count:5d} ({percentage:5.1f}%) - Confidence: {avg_conf:.3f}")
        
        densities = [s['object_density'] for s in scene_analyses]
        print(f"\nScene Statistics:")
        print(f"  Avg objects/image: {total_detections/len(scene_analyses):.1f}")
        print(f"  Avg density: {np.mean(densities):.3f}")
        print(f"  Max objects: {max(s['total_objects'] for s in scene_analyses)}")
        
        self._create_charts(scene_analyses, all_detections, dataset_type)
        
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
    
    def _create_charts(self, scene_analyses, all_detections, dataset_type):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'YOLOv8 Nano - KITTI {dataset_type.upper()} Results', fontsize=16, fontweight='bold')
        
        ax = axes[0, 0]
        times = self.research_metrics['processing_times']
        ax.hist(times, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.1f}ms')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Processing Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        classes = list(self.research_metrics['detection_counts'].keys())
        counts = list(self.research_metrics['detection_counts'].values())
        if classes:
            colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
            ax.bar(classes, counts, color=colors, edgecolor='black')
            ax.set_ylabel('Count')
            ax.set_title('Detections by Class')
            ax.tick_params(axis='x', rotation=45)
        
        ax = axes[1, 0]
        for class_name, confs in self.research_metrics['confidence_scores'].items():
            if len(confs) > 10:
                ax.hist(confs, bins=20, alpha=0.5, label=class_name)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Distribution')
        if self.research_metrics['confidence_scores']:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        sizes = self.research_metrics['detection_sizes']
        if sizes:
            ax.hist(sizes, bins=50, edgecolor='black', alpha=0.7, color='green')
            ax.set_xlabel('BBox Area (pixels²)')
            ax.set_ylabel('Frequency')
            ax.set_title('Object Size Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'analysis_{dataset_type}.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    TRAINING_PATH = r"D:\yolo\dataset\KITTI\training\image_2"
    TESTING_PATH = r"D:\yolo\dataset\KITTI\testing\image_2"
    
    research = KITTIYOLOv8Research(
        training_path=TRAINING_PATH,
        testing_path=TESTING_PATH,
        output_dir=r'D:\yolo\research_results'
    )
    
    print("YOLOv8 Nano - KITTI Complete Analysis Pipeline")
    print("="*60)
    
    print("\n[Step 1/5] Converting KITTI labels to YOLO format...")
    research.convert_kitti_to_yolo()
    
    print("\n[Step 2/5] Training YOLOv8 Nano on KITTI...")
    research.train_model(epochs=50, batch_size=16, img_size=640)
    
    print("\n[Step 3/5] Evaluating model accuracy...")
    metrics = research.evaluate_model('training')
    
    print("\n[Step 4/5] Running inference on training samples...")
    train_scenes, train_detections = research.process_dataset('training', num_samples=100)
    if train_scenes:
        research.generate_report(train_scenes, train_detections, 'training')
    
    print("\n[Step 5/5] Running inference on testing samples...")
    test_scenes, test_detections = research.process_dataset('testing', num_samples=100)
    if test_scenes:
        research.generate_report(test_scenes, test_detections, 'testing')
    
    print("\n" + "="*60)
    print("Complete pipeline finished!")
    print(f"All results saved in: D:\\yolo\\research_results\\")