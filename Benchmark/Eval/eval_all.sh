# 1. 在数据集上评测模型 
python test-CMtMedQA.py \
  --model-dir model/PulseMind-72b \
  --test-path CMtMedQA-test/CMtMedQA_test.json \
  --tar-path output/pulsemind_test_result.csv

# 如果以后要测 MedDiagnose，就把下面这段解开：
# python test-MedDiagnose.py \
#   --model-dir model/PulseMind-72b \
#   --test-path MedDiagnose/MedDiagnose_test.json \
#   --tar-path output/test_result.csv


# 2. 生成 GPT 评测用的对比 prompt 文件
python get_prompt.py \
  --ours_file output/pulsemind_test_result.csv \
  --other_file output/gemini_test_result.csv \
  --output_file output/pulsemind_compare_gemini.csv   


# 3. 调用 GPT-4 / 大模型 作为裁判进行评测
prompt=output/pulsemind_compare_gemini.csv
output=output/pulsemind_compare_gemini_result.csv
temperature=0 

MATRIX_API_BASE="https://api.openai.com" \
MATRIX_API_KEY="Bearer sk-xxxxxxxxxxxxxxx" \
MATRIX_MODEL_ID="gpt-4" \
python -u call_llm.py \
  --prompt_csv_file "${prompt}" \
  --output_file_name "${output}" \
  --use_gpt4 True \
  --temperature "${temperature}"


# 4. 汇总评测结果，生成最终统计输出
python result.py \
  --input-csv-file "${output}"
