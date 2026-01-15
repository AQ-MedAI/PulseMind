import pandas as pd
import re
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="统计 PINS/CORE 对比结果中的 赢/输/平 比例"
    )
    parser.add_argument(
        "--input-csv-file",
        type=str,
        default="compare_internvl_result.csv",
        help="输入 CSV 文件路径（包含 Response 和 Id 列）",
    )
    args = parser.parse_args()

    # 读取存储原始数据的CSV文件
    input_csv_file = args.input_csv_file
    df = pd.read_csv(input_csv_file)

    n = len(df)

    win = 0
    lose = 0
    even = 0

    # 逐行处理CSV文件数据
    for i, row in df.iterrows():
        output = row["Response"]
        ids = row["Id"]
        # 使用正则表达式提取结果。
        result = re.search(r"【(.*?)】", output)

        if result:
            extracted_result = result.group(1)
            if extracted_result == "赢":
                win += 1
            elif extracted_result == "输":
                lose += 1
            elif extracted_result == "平":
                even += 1
        else:
            print(f"{ids} 没有找到匹配的结果。")

    # 注意：如果你想打印真正的百分比，可以乘以 100
    print(f"赢: {win/n:.2f}%")
    print(f"输: {lose/n:.2f}%")
    print(f"平: {even/n:.2f}%")

if __name__ == "__main__":
    main()
