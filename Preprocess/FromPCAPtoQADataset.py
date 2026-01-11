"""
FromPCAPtoQADataset.py
======================
功能：将原始 PCAP 文件转换为问答（QA）训练数据集
核心思想：让模型学习从网络数据包的十六进制表示中提取特定字段值，
         类似于阅读理解任务（给定问题和上下文，提取答案）
"""

from scapy.all import *
import pandas as pd
import random as rnd

# ==============================================================================
# 全局配置参数
# ==============================================================================
NAME_FILE_IN = "mix_protocols_19K.pcap"        # 输入的 PCAP 文件名
NAME_FILE_OUT = "Test_RetrmoreProt_noPayload"  # 输出的 Parquet/CSV 文件名（不含扩展名）

MAX_NUM_QUESTIONS = 20000  # 最多生成的问答对数量，防止内存溢出

# 每个数据包生成问题的随机控制参数
# 范围 [0, 10]，值越大，每个包生成的问题越少
# 例如：RND_NUMBER_Q4PKT=3 表示约 70% 概率继续生成问题
RND_NUMBER_Q4PKT = 3

# 十六进制格式化方式：
# - "every4": 每4个字符一组，如 "4500 003c 1c46"（推荐，平衡可读性和效率）
# - "every2": 每2个字符一组，如 "45 00 00 3c 1c 46"（字节级，最细粒度）
# - "noSpace": 不分隔，如 "4500003c1c46"（最紧凑）
PKT_FORMAT = "every4"

PAYLOAD = False        # 是否保留应用层载荷（False 时只保留协议头部）
RETRIEVAL_ONLY = True  # 是否仅用于特征检索（True 时跳过校验和验证等复杂处理）
# ==============================================================================


def read_pcap_header(input_path):
    """
    读取 PCAP 文件并提取数据包信息
    
    主要流程：
    1. 流式读取 PCAP 文件（避免一次性加载到内存）
    2. 移除载荷（可选）
    3. 匿名化 IP 地址和 TTL
    4. 提取字段值（用于生成答案）
    5. 转换为十六进制字符串（用于生成上下文）
    
    Args:
        input_path (str): PCAP 文件路径
    
    Returns:
        tuple: (list_dict_values, list_dict_hex)
            - list_dict_values: 包含所有数据包字段值的字典列表
            - list_dict_hex: 包含所有数据包十六进制字符串的列表
    """
    # 尝试打开 PCAP 文件
    try:
        pr = PcapReader(input_path)  # 使用流式读取器，内存友好
    except:
        print("file ", input_path, "  error")
        exit(-1)
    
    list_dict_values = []  # 存储所有数据包的字段值字典
    list_dict_hex = []     # 存储所有数据包的十六进制字符串
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
                pkt = pkt["IPv6"]               # 提取 IPv6 层
                pkt = modify_IPv6packets(pkt)   # 匿名化（随机化 IP 和 TTL）
                raw_bytes = bytes(pkt["IPv6"])  # 转换为原始字节
            elif IP in pkt:
                pkt = pkt["IP"]                 # 提取 IP 层
                pkt = modify_IPv4packets(pkt)   # 匿名化
                raw_bytes = bytes(pkt["IP"])    # 转换为原始字节
            
            # 步骤3: 将数据包转换为字典（提取字段值）
            dict_pkt = pkt2dict(pkt)
            
            try:
                # 步骤4: 校验和检查（非检索模式下）
                if not RETRIEVAL_ONLY:
                    dict_pkt, pkt = check_checksum(dict_pkt, pkt)
                # 步骤5: 将字段值转换为十六进制格式
                dict_pkt = convert_hexadecimal(dict_pkt, pkt)
            except:
                continue  # 跳过处理失败的数据包

            # 步骤6: 字节转十六进制字符串
            header_hex = raw_bytes.hex()  # 例如: b'\x45\x00' -> "4500"
            
            # 步骤7: 每2个字符（1个字节）添加一个空格
            string = ""
            for i in range(0, len(header_hex), 2):
                string = string + " " + header_hex[i : i + 2]
            final_hex = string.strip()  # 移除首尾空格，得到 "45 00 00 3c..."
            
            # 步骤8: 载荷信息（如果需要）
            if PAYLOAD and not RETRIEVAL_ONLY:
                success, dict_pkt["last_header3L_byte"], dict_pkt["len_payload"] = (
                    compute_byte_payload(dict_pkt, final_hex, pkt)
                )
            
            # 步骤9: 添加到结果列表
            if not PAYLOAD or RETRIEVAL_ONLY or success:
                list_dict_values.append(dict_normalize(dict_pkt))
                list_dict_hex.append(final_hex)
                j += 1  # 计数器加1
                
        except EOFError:
            # PCAP 文件读取完毕，正常退出
            break
    
    return list_dict_values, list_dict_hex


def dict_normalize(dictionary):
    """
    标准化数据包字典的键名
    
    为什么需要标准化？
    1. 统一不同协议的键名，简化后续处理
    2. IPv6 统一为 "IP"，便于与 IPv4 一起处理
    3. TCP/UDP/ICMP 统一为 "3L"（第三层传输层），便于统一查询
    
    Args:
        dictionary (dict): 原始数据包字段字典
    
    Returns:
        dict: 标准化后的字典
    """
    keys = dictionary.keys()
    
    # 网络层标准化：IPv6 -> IP
    if "IPv6" in keys:
        dictionary["IP"] = dictionary.pop("IPv6")
    
    # 传输层标准化：各协议 -> 3L
    if "ICMP" in keys:
        dictionary["3L"] = dictionary.pop("ICMP")
    if "TCP" in keys:
        dictionary["3L"] = dictionary.pop("TCP")
    if "UDP" in keys:
        dictionary["3L"] = dictionary.pop("UDP")
    
    return dictionary


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


def compute_byte_payload(dict_pkt, hex_string, pkt):
    """
    计算载荷的长度和协议头的最后一个字节
    
    为什么需要这个函数？
    - 在 QA 任务中，可以生成关于载荷长度的问题
    - 最后一个头部字节可以作为边界标识
    
    Args:
        dict_pkt (dict): 包含数据包字段的字典
        hex_string (str): 数据包的十六进制字符串（空格分隔）
        pkt: Scapy 数据包对象
    
    Returns:
        tuple: (success, last_header_byte, payload_length)
            - success (int): 1 表示成功，0 表示失败
            - last_header_byte (str): 头部最后一个字节的十六进制
            - payload_length (int): 载荷长度（字节数）
    """
    hex_list = hex_string.split(" ")  # 将十六进制字符串拆分为字节列表
    
    if TCP in pkt:
        # TCP 头部长度计算
        if IPv6 in pkt:
            # IPv6 固定头部 40 字节 + TCP 头部（dataofs * 4）
            header_len = 40 + int(dict_pkt["TCP"]["dataofs"]) * 4
        else:
            # IPv4 头部（ihl * 4）+ TCP 头部（dataofs * 4）
            header_len = (
                int(dict_pkt["IP"]["ihl"]) * 4 + int(dict_pkt["TCP"]["dataofs"]) * 4
            )
        return 1, hex_list[header_len - 1], len(hex_list[header_len:])

    elif UDP in pkt:
        # UDP 载荷长度 = UDP 总长度 - UDP 头部固定 8 字节
        payload_len = int(dict_pkt["UDP"]["len"]) - 8
        return 1, hex_list[-(payload_len + 1)], payload_len

    elif ICMP in pkt:
        # ICMP 头部长度计算
        if IPv6 in pkt:
            header_len = 40 + int(dict_pkt["TCP"]["dataofs"]) * 4 + 4
        else:
            # IPv4 头部 + ICMP 固定头部 4 字节
            header_len = int(dict_pkt["IP"]["ihl"]) * 4 + 4
        return 1, hex_list[header_len - 1], len(hex_list[header_len:])
    else:
        breakpoint()  # 调试断点：遇到未知协议
        return 0


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
    pkt["IPv6"].src = generate_rnd_IPv6()     # 随机生成源 IPv6 地址
    pkt["IPv6"].dst = generate_rnd_IPv6()     # 随机生成目标 IPv6 地址
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


def check_checksum(dict_pkt, pkt):
    """
    校验和检查与处理
    
    功能：
    1. 检查 IP 校验和是否正确
    2. 随机决定是否使用正确或原始校验和（数据增强）
    3. 记录校验和状态供 QA 任务使用
    
    注意：IPv6 没有头部校验和字段，直接跳过
    
    Args:
        dict_pkt (dict): 包含数据包字段的字典
        pkt: Scapy 数据包对象
    
    Returns:
        tuple: (dict_pkt, pkt)
            - dict_pkt: 更新了 checksum_check 字段的字典
            - pkt: 可能修改了校验和的数据包
    """
    if IPv6 in pkt:
        # IPv6 没有头部校验和字段
        pkt["IPv6"]
        dict_pkt["checksum_check"] = "IPv6"
        return dict_pkt, pkt
    
    elif IP in pkt and pkt["IP"].version == 4:
        ip_pkt = pkt["IP"]
        checksum_real = ip_pkt.chksum  # 保存原始校验和
        
        # 计算正确的校验和
        ip_pkt.chksum = 0x00  # 计算时校验和字段需置零
        list_header_ip = [int(byte) for byte in raw(ip_pkt)][:20]  # 取 IP 头部 20 字节
        calculated_checksum = checksum(list_header_ip)
        
        # 数据增强：50% 概率使用计算的校验和，50% 概率使用原始校验和
        dict_pkt["IP"]["chksum"] = pkt["IP"].chksum = (
            calculated_checksum if rnd.randint(0, 10) >= 5 else checksum_real
        )
        
        # 记录校验和是否正确（可作为 QA 问题）
        dict_pkt["checksum_check"] = (
            "Correct" if pkt["IP"].chksum == calculated_checksum else "Wrong"
        )
    else:
        dict_pkt["checksum_check"] = "Unknown"
    
    return dict_pkt, pkt


def convert_hexadecimal(dict_pkt, pkt):
    """
    将字典中的字段值转换为十六进制格式
    
    为什么转换为十六进制？
    1. 与数据包的十六进制表示保持一致
    2. 模型学习十六进制 <-> 十六进制的映射更自然
    3. 避免十进制和十六进制混用造成的混淆
    
    Args:
        dict_pkt (dict): 包含数据包字段的字典
        pkt: Scapy 数据包对象
    
    Returns:
        dict: 字段值已转换为十六进制的字典
    """
    # ========== 网络层字段转换 ==========
    if IPv6 in pkt:
        # IPv6: hlim（跳数限制）和 fl（流标签）
        dict_pkt["IPv6"]["ttl"] = "{:02x}".format(pkt[IPv6].hlim)
        dict_pkt["IPv6"]["id"] = "{:02x}".format(pkt[IPv6].fl)
    else:
        # IPv4 源地址：192.168.1.1 -> c0.a8.01.01
        dict_pkt["IP"]["src"] = (
            "".join(
                [
                    (
                        str(hex(int(el)))[2:] + "."  # 转十六进制，去掉 "0x" 前缀
                        if len(str(hex(int(el)))[2:]) == 2
                        else "0" + str(hex(int(el)))[2:] + "."  # 不足2位补0
                    )
                    for el in dict_pkt["IP"]["src"].split(".")
                ]
            )
        )[:-1]  # 移除末尾多余的点
        
        # IPv4 目标地址：同上
        dict_pkt["IP"]["dst"] = (
            "".join(
                [
                    (
                        str(hex(int(el)))[2:] + "."
                        if len(str(hex(int(el)))[2:]) == 2
                        else "0" + str(hex(int(el)))[2:] + "."
                    )
                    for el in dict_pkt["IP"]["dst"].split(".")
                ]
            )
        )[:-1]
        
        # TTL 和 ID 字段
        dict_pkt["IP"]["ttl"] = "{:02x}".format(pkt[IP].ttl)
        dict_pkt["IP"]["id"] = "{:02x}".format(pkt[IP].id)

    # ========== 传输层字段转换 ==========
    if TCP in pkt:
        # TCP 各字段转换为十六进制
        dict_pkt["TCP"]["ack"] = "{:02x}".format(int(dict_pkt["TCP"]["ack"]))      # 确认号
        dict_pkt["TCP"]["seq"] = "{:02x}".format(int(pkt[TCP].seq))                # 序列号
        dict_pkt["TCP"]["sport"] = "{:02x}".format(int(pkt[TCP].sport))            # 源端口
        dict_pkt["TCP"]["window"] = "{:02x}".format(int(pkt[TCP].window))          # 窗口大小
        dict_pkt["TCP"]["dport"] = "{:02x}".format(int(pkt[TCP].dport))            # 目标端口
        dict_pkt["TCP"]["chksum"] = "{:02x}".format(int(pkt[TCP].chksum))          # 校验和
    elif UDP in pkt:
        dict_pkt["UDP"]["chksum"] = "{:02x}".format(pkt[UDP].chksum)               # UDP 校验和
        dict_pkt["UDP"]["sport"] = "{:02x}".format(int(pkt[UDP].sport))            # UDP 源端口
    elif ICMP in pkt:
        dict_pkt["ICMP"]["chksum"] = "{:02x}".format(pkt[ICMP].chksum)             # ICMP 校验和
    
    return dict_pkt


def pkt2dict(pkt):
    """
    将 Scapy 数据包对象转换为字典
    
    利用 Scapy 的 show2() 方法获取数据包的文本表示，
    然后解析该文本提取各层各字段的值。
    
    show2() 输出格式示例：
    ###[ IP ]###
      version   = 4
      ihl       = 5
      src       = 192.168.1.1
    ###[ TCP ]###
      sport     = 80
      dport     = 443
    
    Args:
        pkt: Scapy 数据包对象
    
    Returns:
        dict: 嵌套字典，结构为 {layer: {field: value, ...}, ...}
              例如：{'IP': {'src': '192.168.1.1', ...}, 'TCP': {...}}
    """
    packet_dict = {}
    
    # 逐行解析 show2() 的输出
    for line in pkt.show2(dump=True).split("\n"):
        if "###" in line:
            # 检测层标识符，如 "###[ IP ]###"
            if "|###" in line:
                # 子层（嵌套层），如 TCP 选项
                sublayer = line.strip("|#[] ")
                packet_dict[layer][sublayer] = {}
            else:
                # 主层
                layer = line.strip("#[] ")
                packet_dict[layer] = {}
        elif "=" in line:
            # 解析键值对，如 "  src       = 192.168.1.1"
            if "|" in line and "sublayer" in locals():
                # 子层的字段
                key, val = line.strip("| ").split("=", 1)
                packet_dict[layer][sublayer][key.strip()] = val.strip("' ")
            else:
                # 主层的字段
                key, val = line.split("=", 1)
                val = val.strip("' ")
                if val:  # 跳过空值
                    try:
                        packet_dict[layer][key.strip()] = str(val)
                    except:
                        packet_dict[layer][key.strip()] = val
        else:
            continue
    return packet_dict


def clean_df(df):
    """
    清理和重命名 DataFrame 列
    
    功能：
    1. 仅保留 QA 任务需要的字段
    2. 重命名列名，使其更简洁、易懂
    
    字段映射说明：
    - IP.src -> srcIP    : 源 IP 地址
    - IP.dst -> dstIP    : 目标 IP 地址
    - IP.ttl -> IPttl    : TTL 生存时间
    - IP.id  -> IPid     : IP 标识符
    - 3L.sport -> src3L  : 传输层源端口
    - 3L.chksum -> chk3L : 传输层校验和
    - 3L.ack -> 3Lack    : TCP 确认号
    - 3L.seq -> 3Lseq    : TCP 序列号
    - 3L.window -> 3Lwnd : TCP 窗口大小
    
    Args:
        df (pd.DataFrame): 原始 DataFrame
    
    Returns:
        pd.DataFrame: 清理后的 DataFrame
    """
    # QA 任务可用的所有字段
    all_elems = [
        "IP.src",
        "IP.dst",
        "IP.ttl",
        "IP.id",
        "3L.ack",
        "3L.window",
        "3L.sport",
        "3L.seq",
        "3L.chksum",
        "len_payload",
        "last_header3L_byte",
        "checksum_check",
    ]
    
    # 仅保留 DataFrame 中实际存在的字段
    fields_to_maintain = [field for field in all_elems if field in df.columns]
    df = df[fields_to_maintain]
    
    # 重命名列，使其更简洁
    df = df.rename(
        columns={
            "IP.src": "srcIP",
            "IP.dst": "dstIP",
            "3L.chksum": "chk3L",
            "IP.id": "IPid",
            "3L.sport": "src3L",
            "3L.ack": "3Lack",
            "3L.seq": "3Lseq",
            "IP.ttl": "IPttl",
            "3L.window": "3Lwnd",
        }
    )
    return df


def create_dictionary_questions():
    """
    从文件加载 QA 任务的问题字典
    
    文件内容格式（questionsQA.txt）：
    字段名,问题文本
    
    示例：
    srcIP,What is the source IP address?
    dstIP,What is the destination IP address?
    IPttl,What is the TTL value?
    
    为什么使用字典？
    - 字典的 key 是字段名，便于根据字段查找对应问题
    - 实现字段与问题的一一对应关系
    
    Returns:
        dict: {字段名: 问题文本, ...}
    """
    q_dictionary = {}
    with open("./sub/questions_txt/questionsQA.txt", "r") as f:
        for line in f:
            line = line.split(",")        # 按逗号分割
            q_dictionary[line[0]] = line[1]  # key=字段名, value=问题
    return q_dictionary


def main():
    """
    主函数：完整的 QA 数据集生成流程
    
    流程：
    1. 读取 PCAP 文件 -> 字段值字典 + 十六进制字符串
    2. 清理和标准化字段
    3. 加载问题模板
    4. 为每个数据包随机生成多个问答对
    5. 保存为 Parquet 和 CSV 文件
    """
    # ========== 阶段1: 设置随机种子，确保可复现性 ==========
    rnd.seed(43)
    
    # ========== 阶段2: 数据读取 ==========
    print("开始读取 PCAP 文件...")
    list_val, list_hex = read_pcap_header(f"./sub/pcap_files/{NAME_FILE_IN}")
    # list_val: 字段值字典列表，用于生成答案
    # list_hex: 十六进制字符串列表，用于生成上下文
    
    # 将字典列表转换为 DataFrame，并清理
    df_values = pd.json_normalize(list_val)  # 展平嵌套字典
    df_values = clean_df(df_values)           # 清理和重命名列
    
    # 加载问题字典
    quest_dict = create_dictionary_questions()
    
    # ========== 阶段3: 初始化结果容器 ==========
    question = []  # 存储问题列
    context = []   # 存储数据包十六进制列
    answers = []   # 存储答案列
    type_q = []    # 存储问题类型（字段名）
    final_df = pd.DataFrame()
    
    # QA 任务可用的所有字段
    fields = [
        "srcIP",           # 源 IP 地址
        "dstIP",           # 目标 IP 地址
        "chk3L",           # 传输层校验和
        "src3L",           # 源端口
        "IPid",            # IP 标识符
        "IPttl",           # TTL
        "3Lwnd",           # TCP 窗口
        "3Lseq",           # TCP 序列号
        "3Lack",           # TCP 确认号
        "last_header3L_byte",  # 头部最后一字节
        "len_payload",     # 载荷长度
        "checksum_check",  # 校验和状态
    ]
    
    z = 0  # 已创建的 QA 对计数器
    
    # ========== 阶段4: 为每个数据包生成多个问答对 ==========
    print("开始生成问答对...")
    for i in range(len(list_hex)):
        used_index = []  # 记录已使用的字段索引，避免重复
        count = 0        # 尝试计数器，防止无限循环
        
        # 为当前数据包生成多个问答对
        while 1:
            # 随机选择一个字段
            index = random.randint(0, len(fields) - 1)
            
            # 检查字段是否可用
            if (
                fields[index] not in df_values.columns  # 字段不存在
                or index in used_index                   # 字段已使用
                or pd.isna(df_values[fields[index]].iloc[i])  # 字段值为空
            ):
                # 退出条件：所有字段都已使用或尝试次数超限
                if len(used_index) == len(df_values.columns) or count == 100:
                    break
                count += 1
                continue
            
            # 记录已使用的字段
            used_index.append(index)
            
            # 添加问题（根据字段获取对应的问题模板）
            question.append(quest_dict[fields[index]])
            
            # 格式化数据包十六进制
            pkt = list_hex[i].replace(" ", "")  # 先移除所有空格
            if PKT_FORMAT == "every4":
                # 每4个字符添加一个空格
                pkt = "".join(
                    [str(pkt[i : i + 4]) + " " for i in range(0, len(pkt), 4)]
                ).strip()
            elif PKT_FORMAT == "every2":
                # 每2个字符添加一个空格
                pkt = "".join(
                    [str(pkt[i : i + 2]) + " " for i in range(0, len(pkt), 2)]
                ).strip()
            
            # 添加上下文（数据包十六进制）
            context.append(pkt)
            
            # 添加答案（字段值）
            answers.append(f"{df_values[fields[index]].iloc[i]}")
            
            # 添加问题类型（用于分析和评估）
            type_q.append(fields[index])
            
            # 显示进度
            print("Created row N°:", z, end="\r")
            z += 1
            
            # 随机决定是否继续为当前数据包生成问题
            # RND_NUMBER_Q4PKT 越大，继续的概率越小
            if random.randint(0, 10) > RND_NUMBER_Q4PKT:
                break
        
        # 检查是否达到最大问题数
        if z > MAX_NUM_QUESTIONS:
            break
    
    # ========== 阶段5: 组装并保存数据集 ==========
    print("\n组装 DataFrame...")
    final_df["question"] = question    # 问题列
    final_df["context"] = context      # 上下文列（数据包十六进制）
    final_df["answer"] = answers       # 答案列
    final_df["pkt_field"] = type_q     # 问题类型列
    
    # 最终格式：
    # | question                    | context              | answer      | pkt_field |
    # |-----------------------------|----------------------|-------------|-----------|
    # | What is the source IP?      | 4500 003c 1c46...    | c0.a8.01.01 | srcIP     |
    # | What is the TTL value?      | 4500 0028 abcd...    | 40          | IPttl     |
    
    # 保存为 Parquet 格式（高效压缩，推荐用于大数据集）
    final_df.to_parquet(f"../1.Datasets/QA/{NAME_FILE_OUT}.parquet")
    
    # 保存为 CSV 格式（便于查看和调试）
    final_df.to_csv(f"../1.Datasets/QA/{NAME_FILE_OUT}.csv")
    
    print(f"数据集已保存，共 {len(final_df)} 个问答对")


# 程序入口
if __name__ == "__main__":
    main()
