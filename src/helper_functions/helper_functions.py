import torch
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.cuda.amp import autocast
import time
from contextlib import contextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@contextmanager
def timer(name: str):
    """Context manager for timing code blocks"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{name} took {elapsed:.2f} seconds")

class MetricTracker:
    """Efficiently tracks running statistics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)

def create_transforms(input_size: int, zoom_factor: float = 1.0) -> transforms.Compose:
    """Create transformation pipeline"""
    if zoom_factor == 1.0:
        resize_size = (input_size, input_size)
        tfms = [transforms.Resize(resize_size)]
    else:
        resize_size = int(input_size / zoom_factor)
        tfms = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size)
        ]
    
    return transforms.Compose(tfms + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def create_dataloader(
    data_dir: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
    zoom_factor: float = 1.0
) -> DataLoader:
    """Create optimized DataLoader"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    dataset = ImageFolder(
        root=data_dir,
        transform=create_transforms(input_size, zoom_factor)
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

def compute_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,)
) -> List[torch.Tensor]:
    """Compute top-k accuracies efficiently"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    return [
        correct[:k].reshape(-1).float().sum(0) * (100.0 / batch_size)
        for k in topk
    ]

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    log_interval: int = 50
) -> float:
    """Efficient model validation with automatic mixed precision"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    metric_tracker = MetricTracker()
    total_batches = len(dataloader)
    
    with timer("Validation"):
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Use automatic mixed precision
            with autocast():
                outputs = model(images)
                acc = compute_accuracy(outputs, targets)[0]
            
            metric_tracker.update(acc.item(), images.size(0))
            
            if batch_idx % log_interval == 0 or batch_idx == total_batches - 1:
                logger.info(
                    f"Batch [{batch_idx}/{total_batches}] "
                    f"Acc: {metric_tracker.val:.2f}% "
                    f"(Avg: {metric_tracker.avg:.2f}%, "
                    f"Min: {metric_tracker.min:.2f}%, "
                    f"Max: {metric_tracker.max:.2f}%)"
                )
    
    return metric_tracker.avg
