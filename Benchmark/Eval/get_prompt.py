import pandas as pd
import csv
import argparse # 1. 导入argparse库

def main(args):
    """
    主函数，用于读取、处理数据并生成输出文件。
    """
    print(f"INFO: Reading our model data from: {args.ours_file}")
    print(f"INFO: Reading other model data from: {args.other_file}")

    try:
        # 读取CSV数据
        df_ours = pd.read_csv(args.ours_file)
        df_other = pd.read_csv(args.other_file)
    except FileNotFoundError as e:
        print(f"ERROR: File not found. Please check your paths. Details: {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to read CSV files. Details: {e}")
        return

    # 检查当前CSV文件的列名，假设它有 'Response'、'models_response' 和 'Id' 列
    csv_headers = ["编号", "url_x", "gpt4_prompt"]

    rows = []
    print("INFO: Processing rows and generating prompts...")
    
    # 使用 iterrows() 遍历 DataFrame
    for i, (_, row1) in enumerate(df_ours.iterrows()):
        # 确保 df_other 有对应的行
        if i >= len(df_other):
            print(f"WARNING: `other_file` has fewer rows than `ours_file`. Stopping at index {i-1}.")
            break
        row2 = df_other.iloc[i]
 
        try:
            # gt = row1["answer"]  #CMtMedQA
            gt = row1["Response"]  #MedDiagnose
    
            ours = row1["models_response"]
            other = row2["models_response"]
            filename = i 
            
            input_prompt = (
                f"""
                    以下是三段医学问诊记录，请仔细阅读：
                    <真实医生问诊记录>
                    {gt}
                    </真实医生问诊记录>
                    <1号模型生成问诊记录>
                    {ours}
                    </1号模型生成问诊记录>
                    <2号模型生成问诊记录>
                    {other}
                    </2号模型生成问诊记录>
                    请您作为专业临床医生，从以下四个维度比较1号模型与2号模型的表现，并以真实医生的问诊标准为参考：

                    请您作为专业评审，从以下四个维度评估1号模型与2号模型的表现，参考真实医生的诊疗标准：
                    1. 主动性：当患者信息不完整时，能否主动询问关键病史或其他对问诊有帮助的信息
                    2. 准确性：诊断建议是否符合医学规范，是否存在事实错误或过度推断
                    3. 有用性：是否为患者提供清晰、有指导性且实用的帮助，以解决患者的疑虑
                    4. 语言质量：能否精准理解患者诉求，回答是否流畅自然、专业易懂
                    要求：
                    - 评分需体现真实医生与AI表现的差异
                    - 避免笼统评价，需针对具体对话内容
                    - 重点考察医学专业性和实际诊疗价值
                    最终请给出评价结果，形式统一为：1号模型比2号模型表现是：【赢】，【平】，或【输】。
                    """
            )
            new_row = [i, filename, input_prompt]
            rows.append(new_row)
        
        except KeyError as e:
            print(f"ERROR: A required column is missing in row {i}. Column name: {e}. Skipping this row.")
            continue

    # 输出 CSV 文件
    output_csv_file = args.output_file
    try:
        with open(output_csv_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            writer.writerows(rows)
        print(f"\nSUCCESS: CSV 文件已成功保存到 {output_csv_file}")
    except IOError as e:
        print(f"ERROR: Could not write to output file. Details: {e}")


if __name__ == '__main__':
    # 2. 创建一个参数解析器
    parser = argparse.ArgumentParser(description="比较两个模型的输出，并为人工评估生成 prompts。")

    # 3. 添加需要的命令行参数
    parser.add_argument(
        '--ours_file', 
        type=str, 
        required=True, 
        help='我们模型的测试结果CSV文件路径。'
    )
    parser.add_argument(
        '--other_file', 
        type=str, 
        required=True, 
        help='对比模型的测试结果CSV文件路径。'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        required=True, 
        help='生成的用于评估的CSV文件的保存路径。'
    )

    # 4. 解析命令行传入的参数
    args = parser.parse_args()

    # 5. 调用主函数，传入解析后的参数
    main(args)

