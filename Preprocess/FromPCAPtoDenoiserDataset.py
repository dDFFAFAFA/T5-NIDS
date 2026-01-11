"""
FromPCAPtoDenoiserDataset.py
============================
功能：将原始 PCAP 文件转换为去噪器（Denoiser）训练数据集
核心思想：让模型学习重建被破坏的网络数据包，类似于 BERT 的 Masked Language Modeling
"""

from scapy.all import *
import pandas as pd

# ==============================================================================
# 全局配置参数
# ==============================================================================
NAME_FILE_IN = "Train_for_denoiser_450K.pcap"  # 输入的 PCAP 文件名
NAME_FILE_OUT = "Train_for_denoiser_450K"      # 输出的 Parquet 文件名（不含扩展名）

MAX_NUM_QUESTIONS = 450000  # 最多处理的数据包数量（45万个），防止内存溢出

# 十六进制格式化方式：
# - "every4": 每4个字符一组，如 "4500 003c 1c46"（推荐，平衡可读性和效率）
# - "every2": 每2个字符一组，如 "45 00 00 3c 1c 46"（字节级，最细粒度）
# - "noSpace": 不分隔，如 "4500003c1c46"（最紧凑）
PKT_FORMAT = "every4"

PAYLOAD = False  # 是否保留应用层载荷（False时只保留协议头部）
# ==============================================================================


def read_pcap_header(input_path):
    """
    读取 PCAP 文件并处理每个数据包
    
    主要流程：
    1. 流式读取 PCAP 文件（避免一次性加载到内存）
    2. 移除载荷（可选）
    3. 匿名化 IP 地址和 TTL
    4. 转换为十六进制字符串
    
    Args:
        input_path (str): PCAP 文件路径
    
    Returns:
        list: 包含所有数据包十六进制字符串的列表
              格式示例：['45 00 00 3c 1c 46...', '45 00 00 28...', ...]
    """
    # 尝试打开 PCAP 文件
    try:
        pr = PcapReader(input_path)  # 使用流式读取器，内存友好
    except:
        print("file ", input_path, "  error")
        exit(-1)
    
    list_dict_hex = []  # 存储所有数据包的十六进制字符串
    j = 0  # 已处理的数据包计数器
    
    # 循环读取数据包，直到达到最大数量或文件结束
    while j < MAX_NUM_QUESTIONS:
        try:
            print("Processed packet N°:", j, end="\r")  # 实时显示进度（覆盖同一行）
            
            # 读取一个数据包
            pkt = pr.read_packet()
            
            # 步骤1: 移除载荷（如果配置为 False）
            if not PAYLOAD:
                pkt = remove_payload(pkt)
            
            # 步骤2: 根据 IP 版本进行不同处理
            if IPv6 in pkt:
                # IPv6 数据包处理
                pkt = pkt["IPv6"]               # 提取 IPv6 层
                pkt = modify_IPv6packets(pkt)   # 匿名化（随机化 IP 和 TTL）
                raw_bytes = bytes(pkt["IPv6"])  # 转换为原始字节
            elif IP in pkt:
                # IPv4 数据包处理
                pkt = pkt["IP"]                 # 提取 IP 层
                pkt = modify_IPv4packets(pkt)   # 匿名化
                raw_bytes = bytes(pkt["IP"])    # 转换为原始字节
            
            # 步骤3: 字节转十六进制字符串
            header_hex = raw_bytes.hex()  # 例如: b'\x45\x00' -> "4500"
            
            # 步骤4: 每2个字符（1个字节）添加一个空格
            string = ""
            for i in range(0, len(header_hex), 2):
                string = string + " " + header_hex[i : i + 2]
            final_hex = string.strip()  # 移除首尾空格，得到 "45 00 00 3c..."
            
            # 添加到结果列表
            list_dict_hex.append(final_hex)
            j += 1  # 计数器加1
            
        except EOFError:
            # PCAP 文件读取完毕，正常退出
            break
    
    return list_dict_hex


def remove_payload(pkt):
    """
    移除传输层的应用层载荷
    
    为什么移除载荷？
    1. HTTPS 等加密流量的载荷是随机的，无学习价值
    2. 减少数据噪声，聚焦协议头部特征
    3. 减小数据集大小，加快训练
    
    注意：ICMP 不移除，因为其载荷通常不加密（如 ping 的回显数据）
    
    Args:
        pkt: Scapy 数据包对象
    
    Returns:
        pkt: 移除载荷后的数据包
    """
    if TCP in pkt:
        pkt[TCP].remove_payload()  # 移除 TCP 载荷
    elif UDP in pkt:
        pkt[UDP].remove_payload()  # 移除 UDP 载荷
    elif ICMP in pkt:
        pkt = pkt  # ICMP 保持不变
    return pkt


def modify_IPv4packets(pkt):
    """
    匿名化 IPv4 数据包
    
    为什么需要匿名化？
    1. 数据隐私：避免暴露真实 IP 地址
    2. 泛化能力：防止模型记住特定 IP，强制学习协议结构
    3. 数据增强：同一个包每次处理 IP 都不同，增加多样性
    
    Args:
        pkt: Scapy IPv4 数据包对象
    
    Returns:
        pkt: 匿名化后的数据包
    """
    pkt["IP"].src = generate_rnd_IP()      # 随机生成源 IP 地址
    pkt["IP"].dst = generate_rnd_IP()      # 随机生成目标 IP 地址
    pkt["IP"].ttl = random.randint(0, 255) # 随机生成 TTL（生存时间）
    return pkt


def modify_IPv6packets(pkt):
    """
    匿名化 IPv6 数据包
    
    功能同 modify_IPv4packets()，但针对 IPv6 格式
    
    Args:
        pkt: Scapy IPv6 数据包对象
    
    Returns:
        pkt: 匿名化后的数据包
    """
    pkt["IPv6"].src = generate_rnd_IPv6()  # 随机生成源 IPv6 地址
    pkt["IPv6"].dst = generate_rnd_IPv6()  # 随机生成目标 IPv6 地址
    pkt["IPv6"].ttl = random.randint(0, 255)  # 随机生成 TTL
    return pkt


def generate_rnd_IPv6():
    """
    生成随机 IPv6 地址
    
    IPv6 格式：8组，每组4个十六进制字符，用冒号分隔
    示例："2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    
    实现细节：
    - 每组由2个随机字节（0-255）组成
    - 每个字节格式化为2位十六进制（不足补0）
    - 拼接后得到4位十六进制
    
    Returns:
        str: 随机生成的 IPv6 地址
    """
    return ":".join(
        map(
            str,
            (
                # 每组生成：2个随机字节 -> 格式化为十六进制 -> 拼接
                ("{:02x}".format(random.randint(0, 255)))  # 第1个字节，如 "20"
                + ("{:02x}".format(random.randint(0, 255)))  # 第2个字节，如 "01"
                # 拼接结果如 "2001"
                for _ in range(8)  # 重复8次，生成8组
            ),
        )
    )


def generate_rnd_IP():
    """
    生成随机 IPv4 地址
    
    IPv4 格式：4个十进制数（0-255），用点分隔
    示例："192.168.1.100"
    
    实现：生成4个随机数 -> 转字符串 -> 用 '.' 连接
    
    Returns:
        str: 随机生成的 IPv4 地址
    """
    return ".".join(map(str, (random.randint(0, 255) for _ in range(4))))


def create_list_questions():
    """
    从文件加载去噪任务的问题列表
    
    文件内容示例（questionsDenoiser.txt）：
    - "Reconstruct the original packet"
    - "Denoise this network packet"
    - "Fix the corrupted packet header"
    
    为什么需要问题？
    - T5 是 Text-to-Text 模型，需要自然语言指令
    - 问题多样性提高模型泛化能力
    - 随机选择问题实现数据增强
    
    Returns:
        list: 包含所有问题的列表
    """
    list_questions = []
    with open("./sub/questions_txt/questionsDenoiser.txt", "r") as f:
        for line in f:
            list_questions.append(line)
    return list_questions


def main():
    """
    主函数：完整的数据处理流程
    
    流程：
    1. 读取 PCAP 文件 -> 十六进制字符串列表
    2. 加载问题列表
    3. 格式化数据包（根据 PKT_FORMAT）
    4. 随机分配问题
    5. 保存为 Parquet 文件
    """
    # ========== 阶段1: 数据读取 ==========
    print("开始读取 PCAP 文件...")
    list_hex = read_pcap_header(f"./sub/pcap_files/{NAME_FILE_IN}")
    # 结果示例：['45 00 00 3c...', '45 00 00 28...', ...]
    
    quest_list = create_list_questions()
    # 结果示例：['Reconstruct the original packet\n', 'Denoise this...', ...]
    
    # ========== 阶段2: 数据格式化与组装 ==========
    question = []  # 存储问题列
    context = []   # 存储数据包十六进制列
    final_df = pd.DataFrame()
    z = 0  # 已创建的数据行计数器
    
    print("开始格式化数据...")
    for i in range(len(list_hex)):
        # 步骤1: 随机选择一个问题（数据增强）
        index = random.randint(0, len(quest_list) - 1)
        
        # 步骤2: 获取数据包并移除所有空格（统一起点）
        pkt = list_hex[i].replace(" ", "")
        # 从 "45 00 00 3c" -> "4500003c"
        
        # 步骤3: 根据配置的格式重新格式化
        if PKT_FORMAT == "every4":
            # 每4个字符添加一个空格
            # "4500003c1c46" -> "4500 003c 1c46"
            pkt = "".join(
                [str(pkt[i : i + 4]) + " " for i in range(0, len(pkt), 4)]
            ).strip()
            
        elif PKT_FORMAT == "every2":
            # 每2个字符（1个字节）添加一个空格
            # "4500003c" -> "45 00 00 3c"
            pkt = "".join(
                [str(pkt[i : i + 2]) + " " for i in range(0, len(pkt), 2)]
            ).strip()
        # 如果是 "noSpace"，则保持 pkt 不变（无空格）
        
        # 步骤4: 添加到结果列表
        question.append(quest_list[index])  # 添加随机选择的问题
        context.append(pkt)                 # 添加格式化后的数据包
        
        # 显示进度
        print("Created row N°:", z, end="\r")
        z += 1
        
        # 双重保险：防止超过最大数量
        if z > MAX_NUM_QUESTIONS:
            break
    
    # ========== 阶段3: 保存数据集 ==========
    print("\n组装 DataFrame...")
    final_df["question"] = question  # 问题列
    final_df["context"] = context    # 数据包十六进制列
    
    # 最终格式：
    # | question                        | context                  |
    # |---------------------------------|--------------------------|
    # | Reconstruct the original packet | 4500 003c 1c46 4000...   |
    # | Denoise this network packet     | 4500 0028 abcd 4000...   |
    
    output_path = f"../1.Datasets/Denoiser/{NAME_FILE_OUT}.parquet"
    final_df.to_parquet(output_path)
    print(f"数据集已保存到：{output_path}")
    print(f"总共处理了 {len(final_df)} 个数据包")


# 程序入口
if __name__ == "__main__":
    main()
