# -*- coding: utf-8 -*-
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, wait, TimeoutError
import threading
import csv
import random
import os
import select
import sys
import argparse
import pandas as pd
import requests
import json

"""
自包含版本的批量调用脚本：
- 调用 Matrix LLM 网关 (API_BASE / API_KEY / MODEL_ID)
- 从 CSV 批量读取 prompt，并发请求
- 支持重试 / 超时终止 / 断点重跑
"""

# ============================================================
# 配置区：按你们的实际信息改即可
# ============================================================
# https://api.openai.com
API_BASE = os.getenv("MATRIX_API_BASE", "https://api.openai.com")
API_KEY = os.getenv(
    "MATRIX_API_KEY",
    "Bearer sk-xxxxxxxxxxxxxxxx"  # 建议用环境变量覆盖
)
MODEL_ID = os.getenv("MATRIX_MODEL_ID", "gpt-4")
# 如果要换模型，比如 vision 模型，直接改上面 MODEL_ID 即可

# 如果你们的 API 路径不是 /v1/chat/completions，
# 在这里改，比如 "/v1/responses" 等
CHAT_COMPLETION_PATH = "/v1/chat/completions"


# ============================================================
# 参数解析
# ============================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y")


parser = argparse.ArgumentParser(description='批量请求 GPT/MatrixLLM')

parser.add_argument('--is_async',
                    type=str2bool,
                    default=True,
                    help='是否使用异步思路（本脚本内部仍然用线程池并发）')

parser.add_argument('--worker_num',
                    type=int,
                    default=50,
                    help='并发度（同步模式下使用），调太高容易超 token/min 限制')

parser.add_argument('--async_worker_num',
                    type=int,
                    default=600,
                    help='并发度（异步模式下使用），会自动调整')

parser.add_argument('--max_retries',
                    type=int,
                    default=50,
                    help='单条请求最大重试次数')

parser.add_argument('--max_timeout_seconds',
                    type=int,
                    default=90000,
                    help='最长总等待时间，超过后写出已完成的结果并结束')

parser.add_argument('--prompt_csv_file',
                    type=str,
                    default="prompts.csv",
                    help='输入的 prompt CSV 文件')

parser.add_argument('--output_file_name',
                    type=str,
                    default="gpt_response.csv",
                    help='输出结果 CSV 文件名')

parser.add_argument(
    '--is_retry',
    type=str2bool,
    default=False,
    help=
    '断点续跑：如果为 true，会先加载 output file，把其中 Timeout/空值 重新请求'
)

parser.add_argument('--use_gpt4',
                    type=str2bool,
                    default=False,
                    help='保留原有参数，不再使用；是否 gpt4 改成由 MODEL_ID 控制')

parser.add_argument('--temperature',
                    type=float,
                    default=0.0,
                    help='采样温度，0 表示尽量 deterministic')

args = parser.parse_args()

IS_ASYNC = args.is_async
WORKER_NUM = args.async_worker_num if args.is_async else args.worker_num
MAX_RETRIES = args.max_retries
MAX_TIMEOUT_SECONDS = args.max_timeout_seconds
PROMPT_CSV_FILE = args.prompt_csv_file
OUTPUT_FILE_NAME = args.output_file_name
IS_RETRY = args.is_retry
TEMPERATURE = args.temperature

DEBUG = False

# ============================================================
# 读取 prompts
# 约定：CSV 至少包含两列：
#   - gpt4_prompt: 要发给模型的 prompt
#   - url_x:       这里当成一个 ID（你原代码里是 URL，这里不做图像处理，原样写回）
# ============================================================

prompts = []
ids = []
raw_data = pd.read_csv(PROMPT_CSV_FILE).fillna("")

for i in range(len(raw_data)):
    prompt = raw_data["gpt4_prompt"][i]
    if prompt is None or prompt == "":
        continue
    prompts.append(str(prompt))
    ids.append(str(raw_data["url_x"][i]))

# 断点续跑：读取原有输出
responses_cache = []
if IS_RETRY and os.path.exists(OUTPUT_FILE_NAME):
    with open(OUTPUT_FILE_NAME, 'r', encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        next(reader, None)  # 跳过header
        for row in reader:
            # 约定第 3 列是 Prompt，第 4 列是 Response
            if len(row) >= 4:
                responses_cache.append(row[3])
            else:
                responses_cache.append("")


# ============================================================
# 调用 Matrix LLM 的底层函数
# ============================================================

def call_matrix_llm(prompt, temperature=0.0):
    """
    直接调用 Matrix LLM 的 HTTP 接口。
    如果你们接口不是 OpenAI chat/completions 兼容的，
    只要在这里按实际协议改一下 payload/解析逻辑即可。
    """
    url = API_BASE.rstrip("/") + CHAT_COMPLETION_PATH

    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json; charset=utf-8"
    }

    payload = {
        "model": MODEL_ID,
        "temperature": float(temperature),
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # 如果你们还有别的字段，例如 user / extra_params 等，可在这里补充
    # payload["user"] = "评测脚本"

    resp = requests.post(url,
                         headers=headers,
                         data=json.dumps(payload),
                         timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # 假定兼容 OpenAI ChatCompletion 返回格式
    # {
    #   "choices": [
    #       {
    #           "message": {"role": "assistant", "content": "xxx"}
    #       }
    #   ]
    # }
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        # 如果返回结构不一样，可以先打印出来排查
        if DEBUG:
            print("Unexpected response json:", json.dumps(data, ensure_ascii=False))
        raise
    return content


# ============================================================
# 带重试的单条请求封装
# ============================================================

def request_with_try(idx, prompt, id_for_log, cancel_flag, temp=0.0):
    """
    :param idx: 在 prompts 列表中的索引
    :param prompt: 文本 prompt
    :param id_for_log: 这里原来是 url，这里当成标识用，不参与请求
    :param cancel_flag: 用于全局取消
    :param temp: temperature
    """
    # 断点续跑命中缓存
    if IS_RETRY:
        if idx < len(responses_cache):
            cached = responses_cache[idx]
            if cached not in [None, "", "Timeout"]:
                return cached

    retry_count = 0
    result = None

    # 打散启动时间，避免瞬间打满
    time.sleep(random.randint(0, 120))

    while retry_count < MAX_RETRIES:
        if cancel_flag.is_set():
            return None
        try:
            result = call_matrix_llm(prompt, temperature=temp)
            break
        except Exception as e:
            if DEBUG:
                print(
                    f"[Retry {retry_count}] Exception at idx:{idx}, id:{id_for_log}, error:{e}"
                )
            if retry_count == MAX_RETRIES - 1:
                print(
                    f"[MAX_RETRIES] idx:{idx}, id:{id_for_log}, prompt:{prompt[:50]}..., error:{e}, last_result:{result}"
                )
            # 随机 sleep 防止所有线程一起重试
            time.sleep(random.random() * 15 + 3)
            retry_count += 1

    return result


# ============================================================
# 主逻辑：并发执行 + 超时等待 + 写出 CSV
# ============================================================

def main():
    total_cnt = len(prompts)
    print(f"Total prompts: {total_cnt}")
    if total_cnt == 0:
        print("No valid prompts found, exit.")
        return

    cancel_flag = threading.Event()

    with ThreadPoolExecutor(max_workers=WORKER_NUM) as executor:
        start_time = time.time()
        print(f"Start at: {datetime.datetime.now().strftime('%H:%M:%S')}")

        futures = [
            executor.submit(request_with_try, idx, prompt, id_for_log,
                            cancel_flag, TEMPERATURE)
            for idx, (prompt, id_for_log) in enumerate(zip(prompts, ids))
        ]

        total_waited_second = 0
        TIMEOUT_SECONDS_PER_WAIT = 10
        LOG_INTERVAL = 60  # 必须是 TIMEOUT_SECONDS_PER_WAIT 的倍数

        try:
            while total_waited_second < MAX_TIMEOUT_SECONDS:
                # 支持命令行输入 "quit" / "exit" 提前结束
                any_input, _, _ = select.select([sys.stdin], [], [], 0)
                if any_input:
                    user_input = sys.stdin.readline().strip()
                    print(
                        f"User input: {user_input}. 输入 'quit' 或 'exit' 终止程序并写出结果。"
                    )
                    if user_input in ["quit", "exit"]:
                        cancel_flag.set()
                        raise TimeoutError

                # 每 10 秒检查一次
                _, _ = wait(futures, timeout=TIMEOUT_SECONDS_PER_WAIT)
                total_waited_second += TIMEOUT_SECONDS_PER_WAIT

                if total_waited_second % LOG_INTERVAL == 0:
                    finished_count = sum(future.done() for future in futures)
                    print(
                        f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}, "
                        f"Finished: {finished_count}/{total_cnt}"
                    )
                    if finished_count == total_cnt:
                        break
        except TimeoutError:
            print("Timeout or manual stop occurred.")

        # 写出结果
        finished_cnt = 0
        with open(OUTPUT_FILE_NAME, 'w', encoding="utf-8-sig", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', "Id", 'Prompt', 'Response'])

            for idx, future in enumerate(futures):
                prompt = prompts[idx]
                id_for_log = ids[idx]

                if future.done():
                    resp = future.result()
                    if resp is None:
                        resp = "Timeout"
                    writer.writerow([idx, id_for_log, prompt, resp])
                    finished_cnt += 1
                elif IS_RETRY and idx < len(responses_cache):
                    cached = responses_cache[idx]
                    writer.writerow([idx, id_for_log, prompt, cached])
                    finished_cnt += 1
                else:
                    writer.writerow([idx, id_for_log, prompt, "Timeout"])

        end_time = time.time()
        elapsed_time = end_time - start_time
        timeout_num = total_cnt - finished_cnt
        speed = finished_cnt / elapsed_time if elapsed_time > 0 else 0

        print(
            f"Finished num: {finished_cnt}, timeout num: {timeout_num}, "
            f"finished rate: {round(finished_cnt / total_cnt, 2)}"
        )
        print(
            f"Total time: {round(elapsed_time, 2)} seconds, "
            f"process speed: {round(speed, 2)} prompt/sec"
        )

        # 强制杀掉当前进程，结束所有线程
        os.kill(os.getpid(), 9)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，中断任务，安全退出。")
        sys.exit(0)
