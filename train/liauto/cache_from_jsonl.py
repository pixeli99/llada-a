import os
import cv2
import torch
import numpy as np
import argparse
import logging
import json
import gzip
import pickle
import signal
import sys
import torch.multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import warnings
from contextlib import redirect_stdout

from liauto.liauto_dataset import OnlineDataset
from utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """保存数据到gzip压缩的pickle文件"""
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)

def process_sample(data, dataset, cache_path, rank=0):
    """处理单个样本数据"""
    try:
        npz_path = data["npz_path"]
        scene_token = data.get("scene_token", Path(npz_path).stem.split("**")[0])
        
        # 获取原始数据 - 直接传递npz文件路径而不是索引
        raw_data = dataset.prepare_raw_data(npz_path)
        
        # 设置保存路径
        scene_path = cache_path / scene_token / raw_data['log_name']
        os.makedirs(scene_path, exist_ok=True)
        
        # 准备特征和目标
        features = dataset.prepare_features(raw_data)
        feature_path = scene_path / "transfuser_feature.gz"
        dump_feature_target_to_pickle(feature_path, features)
        
        targets = dataset.prepare_targets(raw_data)
        target_path = scene_path / "transfuser_target.gz"
        dump_feature_target_to_pickle(target_path, targets)
        
        return True, scene_token, npz_path
    except Exception as e:
        return False, npz_path, str(e)

def load_jsonl_file(jsonl_path):
    """从JSONL文件加载NPZ路径"""
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "npz_path" not in record:
                logger.warning(f"JSONL记录中缺少'npz_path'字段: {line}")
                continue
            records.append(record)
    return records

def process_worker(jsonl_path, cache_path, rank, num_workers, jwt_token=None):
    """工作进程处理函数"""
    # 加载JSONL记录
    all_records = load_jsonl_file(jsonl_path)
    logger.info(f"从JSONL加载了 {len(all_records)} 条记录")
    
    # 分配工作负载
    num_records = len(all_records)
    chunk_size = num_records // num_workers
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != num_workers-1 else num_records
    worker_records = all_records[start_idx:end_idx]
    
    logger.info(f"进程 {rank}: 处理 {len(worker_records)} 个记录 ({start_idx} 到 {end_idx-1})")
    
    # 创建数据集实例(只用于数据处理，不用于迭代)
    val_config_path = "/lpai/volumes/ad-vla-vol-ga/lipengxiang/lipx_dev/TrajHF/liauto/projects/end2end/all_in_one_val.py"
    val_cfg = Config.fromfile(val_config_path)
    dataset = OnlineDataset(
        dataset_config=val_cfg.txt_root,  # 不需要配置，因为我们直接处理NPZ文件
        jsonl_npz_file="/lpai/volumes/ad-vla-vol-ga/lipengxiang/lipx_dev/TrajHF/defensive_45k_0421.jsonl",  # 不需要从JSONL文件加载，因为我们已经加载了记录
    )
    
    # 如果提供了JWT令牌，则使用它
    if jwt_token:
        dataset.jwt_token = jwt_token
    
    # 处理分配的记录
    success_count = 0
    fail_count = 0
    
    pbar = tqdm(total=len(worker_records), desc=f"进程 {rank}", position=rank)
    
    for record in worker_records:
        result, token_or_path, msg = process_sample(record, dataset, cache_path, rank)
        if result:
            success_count += 1
            pbar.set_description(f"进程 {rank}: 成功 {success_count}")
        else:
            fail_count += 1
            logger.warning(f"进程 {rank}: 处理文件失败 {token_or_path}: {msg}")
            pbar.set_description(f"进程 {rank}: 成功 {success_count}, 失败 {fail_count}")
        pbar.update(1)
    
    pbar.close()
    logger.info(f"进程 {rank} 完成: {success_count} 成功, {fail_count} 失败")
    return success_count, fail_count

def parse_args():
    parser = argparse.ArgumentParser(description="从JSONL文件创建数据缓存")
    parser.add_argument("--jsonl_file", type=str, required=True, 
                       help="包含NPZ文件路径的JSONL文件")
    parser.add_argument("--cache_path", type=str, default="/lpai/volumes/ad-vla-vol-ga/lipengxiang/liauto_jsonl_data_cache",
                       help="缓存输出路径")
    parser.add_argument("--num_workers", type=int, default=32,
                       help="工作进程数量")
    parser.add_argument("--jwt_token", type=str, default=None,
                       help="用于资源访问的JWT令牌")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.jsonl_file):
        logger.error(f"JSONL文件不存在: {args.jsonl_file}")
        return
    
    cache_path = Path(args.cache_path)
    cache_path.mkdir(exist_ok=True, parents=True)
    
    jwt_token = args.jwt_token
    if jwt_token is None:
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJ3ZW54aW4zQGxpeGlhbmcuY29tIiwiaXNzIjoibHBhaSIsImlhdCI6MTY0NzU3MzkwNywianRpIjoiOWNiNzc0NjQtZGVjYS00NTRhLWIyZTEtYmI4NTA2N2VjM2JhIn0.Krz29HMzS6z762FsxjSUbbNJ0qRchqmtPM3FK2lrXrs"
    
    logger.info(f"开始处理JSONL文件: {args.jsonl_file}")
    logger.info(f"缓存将保存到: {cache_path}")
    logger.info(f"使用 {args.num_workers} 个工作进程")
    
    if args.num_workers > 1:
        # 多进程处理
        processes = []
        
        def signal_handler(signum, frame):
            print("\n正在清理进程...")
            for p in processes:
                p.terminate()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            for rank in range(args.num_workers):
                p = mp.Process(
                    target=process_worker,
                    args=(args.jsonl_file, cache_path, rank, args.num_workers, jwt_token)
                )
                p.start()
                processes.append(p)
                
            for p in processes:
                p.join()
                
        except KeyboardInterrupt:
            print("\n正在清理进程...")
            for p in processes:
                p.terminate()
            sys.exit(0)
    else:
        # 单进程处理
        process_worker(args.jsonl_file, cache_path, 0, 1, jwt_token)
    
    logger.info(f"数据处理完成! 缓存保存在: {cache_path}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
