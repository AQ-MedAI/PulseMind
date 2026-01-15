# -*- encoding: utf-8 -*-
import os
import torch
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import json
import logging
import time
import pandas as pd
import re
import argparse  # 新增

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_HOST_IP"] = os.environ["POD_IP"]

logger = logging.getLogger()

def read_jsonl_to_dataframe(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            # 逐行解析 JSON 对象
            json_data = json.loads(line.strip())
            data.append(json_data)

    # 将数据转换为 pandas 数据框
    df = pd.DataFrame(data)
    return df

# 用户自定义代码处理类


class UserHandler:
    """
    model_dir: model.py 文件所在目录
    """

    def __init__(self, model_dir):
        MODEL_PATH = model_dir
        self.llm = LLM(
            model=MODEL_PATH,
            limit_mm_per_prompt={"image": 30},
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            enforce_eager=False,
            # use_v2_block_manager=True,
            disable_custom_all_reduce=True,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_k=1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=2048,
            stop_token_ids=[],
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

    """
     测试demo
     1 输入配置
        query:TYPE_STRING:[1]        对应bytes 类型
     2 输出配置
        out_float:TYPE_FP64:[1]        对应float64 类型
        out_string:TYPE_STRING:[1]     对应bytes_ 类型
        out_int:TYPE_INT32:[1].        对应int32 类型
     其他
     1. TYPE_STRING对应是python bytes类型(或者np.bytes_)，目标是方便传递二进制内容，比如图片的binary内容，减少base64转换开销;
        bytes类型可以通过decode函数明确转换成python str类型
     2. 参数维度见使用文档
    """

    def predict_np(self, features, trace_id):
        # 解析请求, 字符串类型默认是bytes类型
        resultCode = 0
        errorMessage = "ok"
        trace_id = "none"
        try:
            start = time.time()
            query = features.get("query").decode()
            query_body = json.loads(query)
            llm_inputs = list()
            trace_id = query_body.get("trace_id", "")
            logger.info(f"trace_id: {trace_id}")
            
            messages = query_body["messages"]
           
            image_inputs, video_inputs = process_vision_info(messages)
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            llm_inputs_ = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            llm_inputs.append(llm_inputs_)
            logger.info(f"开始生成前耗时：{time.time() - start}")
            outputs = self.llm.generate(llm_inputs,
                                        sampling_params=self.sampling_params)
            generated_ids = [list(outputs[0].outputs[0].token_ids)]
            generated_text_complete = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False)[0]
            generated_text = generated_text_complete.replace("<|im_end|>", "")
            resultMap = {"generated_text": generated_text}
            logger.info(f"{resultMap}")
            logger.info(f"整体耗时: {str(time.time() - start)}")
            # 处理结果返回
            resultCode = 0  # 0表示成功，其它为失败
            errorMessage = "ok"  # errorMessage为predict函数对外透出的信息
            logger.info("predict result")  # 可以在平台查看相关日志
            return (resultCode, errorMessage, resultMap)
        except Exception as e:
            ans = "{} infer error {}".format(trace_id, str(e))
            logger.error(ans)
            resultMap = {"generated_text": ans}
            return (resultCode, errorMessage, resultMap)

        


if __name__ == "__main__":
    # ===== 1. 解析命令行参数 =====
    parser = argparse.ArgumentParser(description="Run PulseMind-72b on CMtMedQA_test")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="PulseMind-72b",
        help="模型所在目录（即原来的 MODEL_PATH）",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="CMtMedQA_test.json",
        help="输入 JSON 测试文件路径",
    )
    parser.add_argument(
        "--tar-path",
        type=str,
        default="test_result.csv",
        help="输出 CSV 结果文件路径",
    )

    args = parser.parse_args()

    # ===== 2. 用命令行参数替换原来的硬编码变量 =====
    user_handler = UserHandler(args.model_dir)
    test_path = args.test_path
    tar_path = args.tar_path

    # 读取JSON文件并转换为DataFrame
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data[:1000] #前1000例才有历史对话

    # 假设JSON是一个列表，每个元素是一个对象（即原来的行）
    df = pd.DataFrame(data)

    models_response = list()
    real_answers = list()
    for index, row in df.iterrows():
        print(row)
        messages = list()
        real_answer = list()
        questions = row["instruction"]
        label = row["output"]
        cate1 = row["cate1"]
        cate2 = row["cate2"]
        history = row["history"]
        system = f"你现在是一名医生。一位患者会向你咨询一些问题。请利用你专业的医学知识直接回答患者的问题。"
        messages.append({"role": "system", "content": system})
        try:
            for communication in history:
                m = {"role": "user", "content": []}
                question = communication[0]
                answer = communication[1]
                m["content"].append({"type": "text", "text": question})
                messages.append(m)
                real_answer.append(m)
                real_answer.append({"role": "doctor", "content": answer})
                try:
                    # Call the model with the image URL and the prompt
                    request = {}
                    query_body = dict()
                    query_body["messages"] = messages
                    query_body["trace_id"] = "xj_test"
                    request["query"] = json.dumps(query_body).encode()
                    ans = user_handler.predict_np(
                        request, "test_trace_id")[2]["generated_text"]
                    print(ans)
                    m = {"role": "assistant", "content": ans}
                    messages.append(m)
                    print(messages)
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    messages.append({"role": "assistant", "content": "Error fetching response"})


        except Exception as e:
            print(f"Error processing :{e}")

        m = {"role": "user", "content": []}
        m["content"].append({"type": "text", "text": questions})
        messages.append(m)
        real_answer.append(m)
        real_answer.append({"role": "doctor", "content": label})

        request = {}
        query_body = dict()
        query_body["messages"] = messages
        query_body["trace_id"] = "xj_test"
        request["query"] = json.dumps(query_body).encode()
        ans = user_handler.predict_np(
            request, "test_trace_id")[2]["generated_text"]
        print(ans)
        m = {"role": "assistant", "content": ans}
        messages.append(m)
        print(messages)

        models_response.append(json.dumps(messages, ensure_ascii=False))
        real_answers.append(json.dumps(real_answer, ensure_ascii=False))

    df["models_response"] = models_response
    df["answer"] = real_answers
    df.to_csv(tar_path)
                
