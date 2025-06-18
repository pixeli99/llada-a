# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import os.path as osp
import json
import argparse
import datetime
import time
from multiprocessing import Manager, Process, Queue, cpu_count
import threading
from typing import List, Any, Tuple, Set

import pandas as pd
from tqdm import tqdm
import colorama
from colorama import Fore, Style
import logging

# --------------------------------- 日志设置 ---------------------------------
colorama.init()
logging.basicConfig(
    level=logging.INFO,
    format=f'{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - '
           f'{Fore.BLUE}%(levelname)s{Style.RESET_ALL} - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ------------------------------ 工具函数 -------------------------------------

def load_target_tokens(json_path: str) -> Set[str]:
    """加载目标 scene_token 集合。JSON 需形如 [{"scene_token": "..."}, ...]."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        tokens = {item['scene_token'] for item in data if 'scene_token' in item}
        logger.info(f"{Fore.GREEN}加载 {len(tokens)} 个唯一 scene_token{Style.RESET_ALL}")
        return tokens
    except Exception as e:
        logger.error(f"{Fore.RED}读取/解析 tokens JSON 出错: {e}{Style.RESET_ALL}")
        return set()

def extract_scene_token(scene_label_path: str) -> str:
    """从 path A/B/clip_success.txt 提取中间的 token (index==1)."""
    parts = scene_label_path.strip().split('/')
    return parts[1] if len(parts) >= 2 else ''

# ----------------------------- 多进程核心 ------------------------------------

def process_batch(batch: List[str], target_tokens: Set[str],
                  result_q: Queue, prog_q: Queue, pid: int) -> None:
    """子进程：批量处理 .csv 文件列表。"""
    token_seen: Set[str] = set()
    local_res: List[Tuple[Any, Any, Any]] = []

    for csv_path in batch:
        matched = 0
        try:
            df = pd.read_csv(csv_path, header=None, usecols=[0, 1, 2])
        except Exception:
            prog_q.put((pid, 1, 0))
            continue
        base_path = csv_path.split('/')[:-2]
        base_path = '/'.join(base_path) + '/'
        for scene_path, num_f, num_l in df.itertuples(index=False):
            token = extract_scene_token(scene_path)
            if token and token not in token_seen and token in target_tokens:
                local_res.append((os.path.join(base_path, scene_path), num_f, num_l))
                token_seen.add(token)
                matched += 1

        prog_q.put((pid, 1, matched))

    result_q.put(local_res)

def distribute_tasks(items: List[str], n_proc: int) -> List[List[str]]:
    batches = [[] for _ in range(n_proc)]
    for idx, item in enumerate(items):
        batches[idx % n_proc].append(item)
    return batches

# --------------------------- 增量保存 & 监控 ---------------------------------

def save_results(df: pd.DataFrame, path: str) -> bool:
    try:
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        logger.error(f"{Fore.RED}最终保存失败: {e}{Style.RESET_ALL}")
        return False

def incremental_save_worker(res_q: Queue, save_q: Queue, final_path: str,
                             interval: int = 60, keep_inc: bool = True):
    all_rows = []
    last_t = time.time()
    inc_id = 0
    inc_files = []
    dir_ = osp.dirname(final_path) or '.'
    base = osp.splitext(osp.basename(final_path))[0]

    while True:
        try:
            batch = res_q.get(timeout=1)
        except Exception:
            batch = None
        if batch == "DONE":
            break
        if batch:
            all_rows.extend(batch)
            save_q.put(len(all_rows))

        if time.time() - last_t >= interval and all_rows:
            inc_id += 1
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            tmp_path = (f"{dir_}/{base}_inc{inc_id}_{ts}.csv" if keep_inc
                        else f"{final_path}.temp")
            pd.DataFrame(all_rows, columns=['scene_label_path', 'num_frames', 'num_frames_labeled']) \
              .drop_duplicates('scene_label_path').to_csv(tmp_path, index=False)
            last_t = time.time()
            save_q.put(f"SAVED:{json.dumps({'count': len(all_rows), 'path': tmp_path, 'timestamp': ts})}")
            if keep_inc:
                inc_files.append(tmp_path)

    # 完成后保存最终结果
    ok = False
    if all_rows:
        df = pd.DataFrame(all_rows, columns=['scene_label_path', 'num_frames', 'num_frames_labeled'])
        df.drop_duplicates('scene_label_path', inplace=True)
        ok = save_results(df, final_path)
    save_q.put(f"FINAL:{json.dumps({'status': 'DONE' if ok else 'ERROR', 'incremental_files': inc_files})}")

def monitor_progress(prog_q: Queue, total: int, save_q: Queue, t0: float):
    done = 0
    matched = 0
    inc_info = []
    with tqdm(total=total, desc='处理进度') as bar:
        while done < total:
            try:
                pid, f_cnt, m_cnt = prog_q.get(timeout=0.1)
                done += f_cnt
                matched += m_cnt
                bar.update(f_cnt)
                if done and done % 100 == 0:
                    elapsed = time.time() - t0
                    h, rem = divmod(elapsed, 3600)
                    m, s = divmod(rem, 60)
                    logger.info(f"已处理 {done}/{total}，匹配 {matched}，耗时 {int(h):02d}:{int(m):02d}:{int(s):02d}")
            except Exception:
                pass
            try:
                msg = save_q.get_nowait()
                if isinstance(msg, int):
                    continue
                if msg.startswith('SAVED:'):
                    info = json.loads(msg[6:])
                    inc_info.append((info['timestamp'], info['count'], info['path']))
                    logger.info(f"{Fore.CYAN}增量保存 → {info['path']}，{info['count']} 条{Style.RESET_ALL}")
                elif msg.startswith('FINAL:'):
                    break
            except Exception:
                pass
            time.sleep(0.01)
# ------------------------------- 入口 ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Filter scenes by scene_token fast.')
    ap.add_argument('--tokens', required=True, help='JSON 文件，包含 scene_token 列表')
    ap.add_argument('--labels', nargs='*', default=[], help='一个或多个 CSV 标注文件')
    ap.add_argument('--label-file', help='txt 文件，每行一个 CSV 路径')
    ap.add_argument('--output', required=True, help='输出 CSV 路径')
    ap.add_argument('--nproc', type=int, default=None, help='进程数，默认 = CPU 核心数')
    ap.add_argument('--keep-inc', action='store_true', help='保留所有增量保存文件')
    ap.add_argument('--interval', type=int, default=60, help='增量保存间隔秒')
    args = ap.parse_args()

    # label 文件集合
    label_files: List[str] = list(args.labels)
    if args.label_file:
        try:
            with open(args.label_file, 'r') as lf:
                label_files.extend([l.strip() for l in lf if l.strip()])
        except Exception as e:
            logger.error(f"{Fore.RED}读取 label-file 失败: {e}{Style.RESET_ALL}")
            sys.exit(1)
    if not label_files:
        logger.error(f"{Fore.RED}未提供任何 CSV 标注文件！{Style.RESET_ALL}")
        sys.exit(1)

    # tokens
    tokens = load_target_tokens(args.tokens)
    if not tokens:
        sys.exit(1)

    n_proc = args.nproc or cpu_count()
    logger.info(f"使用 {n_proc} 进程处理 {len(label_files)} 个 CSV…")

    # 队列 & 进程
    mgr = Manager()
    res_q, prog_q, save_q = mgr.Queue(), mgr.Queue(), mgr.Queue()

    batches = distribute_tasks(label_files, n_proc)
    saver = threading.Thread(target=incremental_save_worker,
                             args=(res_q, save_q, args.output, args.interval, args.keep_inc),
                             daemon=True)
    saver.start()

    procs: List[Process] = []
    for i, batch in enumerate(batches):
        if not batch:
            continue
        p = Process(target=process_batch, args=(batch, tokens, res_q, prog_q, i))
        p.start()
        procs.append(p)

    t0 = time.time()
    monitor_progress(prog_q, len(label_files), save_q, t0)

    for p in procs:
        p.join()
    res_q.put("DONE")
    saver.join()

    elapsed = time.time() - t0
    h, rem = divmod(elapsed, 3600); m, s = divmod(rem, 60)
    logger.info(f"{Fore.GREEN}全部完成！耗时 {int(h):02d}:{int(m):02d}:{int(s):02d}{Style.RESET_ALL}")

if __name__ == '__main__':
    main()
