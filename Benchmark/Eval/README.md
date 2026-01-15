# Benchmark 评测说明

本仓库用于展示 **Pulsemind 与其它模型（如 Gemini）在医学场景上的 Benchmark 评测流程**。  
我们将大模型（如 GPT-4）作为裁判，对不同模型在同一测试集上的输出进行对比打分。

> ✅ **快速使用：只需一条命令完成全部评测流程：**
>
> ```bash
> bash eval_all.sh
> ```
>
> `eval_all.sh` 会自动依次完成：模型测试 → 生成对比用 prompt → 调用大模型评测 → 汇总统计结果。

下面是完整的手动执行流程及每一步的作用说明。

---

## 1. 在数据集上评测模型

```bash
python test-CMtMedQA.py
# python test-MedDiagnose.py
```

## 2. 生成对比用的prompt

```bash
python get_prompt.py
```

## 3. 调用 GPT-4 评测

```bash
python call_llm.py
```

## 4. 汇总统计结果

```bash
python result.py
```