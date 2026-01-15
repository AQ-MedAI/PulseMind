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
import argparse  

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_HOST_IP"] = os.environ["POD_IP"]

# 统一继承MayaBaseHandler，可以自动获取组件面板的模型文件

# 组件使用文档，详见 https://yuque.antfin-inc.com/aii/aistudio/nkyse5
# python lib 依赖写入 requirement.txt

logger = logging.getLogger()

# 用户自定义代码处理类


class UserHandler:
    """
    model_dir: model.py 文件所在目录
    """

    def __init__(self, model_dir):
        # 可以认为 self.resource_path 就是上游组件输入的模型或者python组件面板设置的 '自定义资源地址'在本地磁盘的路径，如果都没有返回 None
        MODEL_PATH = model_dir
        self.llm = LLM(
            model=MODEL_PATH,
            limit_mm_per_prompt={"image": 30},
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            enforce_eager=False,
            use_v2_block_manager=True,
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
        # self.processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)

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
                "prompt":prompt,
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

    df = pd.read_csv(test_path)
    models_response = list()
    for index, row in df.iterrows():
        print(row)
        messages = list()
        mapping = row["Id"]
        system = "你现在是一名医生。一位患者会向你咨询一些问题。请利用你专业的医学知识直接回答患者的问题。"
        messages.append({"role": "system", "content": system})
        modified_content = row["Response"].split("\n")
        all_numer = len(modified_content)
        for index, message in enumerate(modified_content):
            if ((index == all_numer - 1 and message[:2] == "患者")
                    or (index % 2 == 0 and message[:2] == "医生")
                    or (index % 2 == 1 and message[:2] == "患者")):
                continue
            if message[:2] == "患者":
                m = {"role": "user", "content": []}
                content = message[3:]
                content_match = re.findall(r"<attachment>(.*?)<attachment>",
                                           content)
                for ids in content_match:
                    ids_list = ids.split(",")
                    count = 0
                    try:
                        image_url = f"/images/{mapping}"

                        m["content"].append({
                            "type": "image",
                            "image": image_url,
                            "max_pixels": 4816896
                        })
                        count += 1
                    except:
                        continue
                    content = content.replace(f"<attachment>{ids}<attachment>",
                                              "")
                m["content"].append({"type": "text", "text": content})
                messages.append(m)
                request = {}
                query_body = dict()
                query_body["messages"] = messages
                query_body["trace_id"] = "wenshuo_test"
                request["query"] = json.dumps(query_body).encode()
                ans = user_handler.predict_np(
                    request, "test_trace_id")[2]["generated_text"]
                print(ans)
                m = {"role": "assistant", "content": ans}
                messages.append(m)
        print(messages)
        models_response.append(json.dumps(messages, ensure_ascii=False))
    df["models_response"] = models_response
    df.to_csv(tar_path)
