"""
此脚本用于下载 CIC-IDS2017 数据集的 Payload-Bytes 子集。
为了节省磁盘空间和时间，它采用“按需下载”策略：
1. 逐个下载数据分片（File 1, 2, 3...）。
2. 每下载完一个分片，立即检查其中的标签（attack_label）。
3. 一旦发现分片中包含多种类别（例如：既有正常流量 BENIGN，又有攻击流量 FTP-Patator），则停止下载。
4. 这样可以获得一个“最小且有效”的数据集用于后续的分类模型评测。
"""

import os
import shutil
from pathlib import Path
from typing import Iterable, Set, Tuple

import pyarrow.parquet as pq
from nids_datasets import Dataset


def has_multiclass_labels(parquet_path: Path, min_classes: int = 2, max_rows: int = 200_000) -> Tuple[bool, Set[str]]:
	"""
	检查指定的 Parquet 文件中是否包含多种攻击标签。
	
	参数:
		parquet_path: Parquet 文件的路径。
		min_classes: 目标类别数量，默认为 2（即至少包含一种攻击和一种正常流量）。
		max_rows: 最大扫描行数，防止在大文件上耗时过长。
		
	返回:
		(是否达到目标类别数, 已发现的标签集合)
	"""
	# 使用 pyarrow 逐块读取，避免一次性加载数 GB 的文件到内存
	pf = pq.ParquetFile(parquet_path)
	seen: Set[str] = set()
	rows_scanned = 0
	
	# 遍历文件内的行组 (Row Groups)
	for rg_idx in range(pf.num_row_groups):
		# 仅读取 attack_label 这一列
		table = pf.read_row_group(rg_idx, columns=["attack_label"])
		labels = table.column("attack_label").to_pylist()
		
		for lbl in labels:
			if lbl is not None:
				seen.add(lbl)
				# 如果已经发现足够多的类别，提前结束扫描
				if len(seen) >= min_classes:
					return True, seen
		
		rows_scanned += table.num_rows
		if rows_scanned >= max_rows:
			break
			
	return len(seen) >= min_classes, seen


def download_payload_bytes(files: Iterable[int], target_dir: Path) -> Tuple[Path, Set[str]]:
	"""
	按顺序下载分片，发现多类标签即停止。

	用最小集合覆盖攻击标签，避免一次性下载 18 个分片占满磁盘。

	返回：找到多类的 parquet 路径和已看到的标签集合；若全程未找到则返回最后一个下载的文件和标签集合。
	"""

	payload_dir = target_dir / "CIC-IDS2017" / "Payload-Bytes"
	seen_labels: Set[str] = set()
	found_path: Path = None  # type: ignore

	for i in files:
		print(f"\n正在尝试下载分片 {i}...")
		# 使用 nids-datasets 库下载指定分片
		ds = Dataset(dataset="CIC-IDS2017", subset=["Payload-Bytes"], files=[i])
		ds.download()

		if not payload_dir.exists():
			raise FileNotFoundError(f"下载完成但未找到目录: {payload_dir}")

		parquet_path = payload_dir / f"Payload_Bytes_File_{i}.parquet"
		if not parquet_path.exists():
			print(f"分片缺失，跳过: {parquet_path}")
			continue
		
		# 检查该分片是否满足多类要求
		ok, labels = has_multiclass_labels(parquet_path)
		seen_labels.update(labels)
		print(f"检查分片 {i}: 标签={labels}")
		
		if ok:
			found_path = parquet_path
			print(f"已找到含多类标签的分片，停止下载后续分片: {parquet_path}")
			break

	if found_path is None:
		# 没找到多类，也返回最后一个存在的文件供用户参考
		existing = sorted(payload_dir.glob("Payload_Bytes_File_*.parquet"))
		if not existing:
			raise FileNotFoundError("未下载到任何 Payload-Bytes 分片")
		found_path = existing[-1]
	return found_path, seen_labels


def main() -> None:
	# 安全检查：防止意外下载大型数据集
	print("=" * 70)
	print("⚠️  警告：此脚本会下载大量数据（每个分片可能数 GB）！")
	print("=" * 70)
	print()
	print("如果只是想验证流程可行，请使用最小化演示脚本：")
	print("  python demo_minimal_data.py      # 生成最小测试数据")
	print("  python demo_pipeline.py          # 运行端到端演示")
	print()
	print("如果确实需要下载完整数据集，请输入 'YES' 确认：")
	
	confirmation = input(">>> ").strip()
	if confirmation != "YES":
		print("\n已取消下载。建议使用 demo_minimal_data.py 进行测试。")
		return
	
	print("\n开始下载数据集...")
	print()
	
	# 1. 确定下载路径：统一存放在项目根目录下的 data/ 文件夹中
	repo_root = Path(__file__).resolve().parent.parent  # nids/encoder/PCAP_encoder
	data_dir = repo_root / "data"
	data_dir.mkdir(exist_ok=True)
	
	# 切换工作目录到 data/，方便 nids-datasets 默认下载
	os.chdir(data_dir)

	# 2. 清理旧数据：为了确保从头开始寻找最小集，先删除旧的 Payload-Bytes 目录
	payload_dir = data_dir / "CIC-IDS2017" / "Payload-Bytes"
	if payload_dir.exists():
		print(f"清理旧的 Payload-Bytes 目录: {payload_dir}")
		shutil.rmtree(payload_dir)

	# 3. 执行下载逻辑：尝试下载 1 到 18 号分片，直到找到含攻击样本的分片
	candidate_files = list(range(1, 19))
	found_path, labels = download_payload_bytes(candidate_files, data_dir)

	print(f"\n完成。可用于评测的 parquet: {found_path}")
	print(f"已观测到的标签集合: {labels}")
	print("如需更小的子集，可手动删除多余分片，仅保留含攻击的那个文件。")


if __name__ == "__main__":
	main()

