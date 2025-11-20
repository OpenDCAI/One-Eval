import pandas as pd
import json
import os

def clean_value(value):
    """清理单个值，处理特殊字符串和类型"""
    if pd.isna(value) or value == '<None>':
        return None
    return value

def parse_list_string(value, separator):
    """
    将包含分隔符的字符串解析为列表。
    能够同时处理全角和半角分隔符。
    """
    if not isinstance(value, str):
        return value
    
    # 将字符串中的全角分隔符统一替换为半角分隔符
    normalized_value = value
    if separator == ';':
        normalized_value = normalized_value.replace('；', ';') # 处理全角分号

    items = [item.strip() for item in normalized_value.split(separator)]
    
    cleaned_items = [clean_value(item) for item in items]
    
    return cleaned_items if cleaned_items != [None] else None


def convert_excel_to_json(excel_path, sheet_name, output_path, primary_key_col='bench_name'):
    """
    读取 Excel 文件，将其转换为结构化的 JSON 文件。

    :param excel_path: 输入的 Excel 文件路径。
    :param sheet_name: 要读取的工作表名称。
    :param output_path: 输出的 JSON 文件路径。
    :param primary_key_col: 用作 JSON 主键的列名。
    """
    print(f"正在读取 Excel 文件: {excel_path} (Sheet: {sheet_name})")

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=str)
        df.replace('<None>', None, inplace=True)
        df = df.where(pd.notna(df), None)

    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {excel_path}")
        return
    except Exception as e:
        print(f"读取 Excel 文件时发生错误: {e}")
        return

    list_columns = {
        'aliases': ';',
        'test_subset': ',',
        'test_split': ',',
        'default_question_split': ',',
        'default_answer_split': ','
    }

    records = df.to_dict(orient='records')
    
    final_json_data = {}

    for record in records:
        key = record.get(primary_key_col)
        if not key:
            print(f"警告: 发现一条记录缺少主键 '{primary_key_col}'，已跳过。")
            continue

        processed_record = {}
        for col, value in record.items():
            if col == primary_key_col:
                continue

            cleaned_val = clean_value(value)

            if col in list_columns and cleaned_val:
                processed_record[col] = parse_list_string(cleaned_val, list_columns[col])
            else:
                processed_record[col] = cleaned_val
        
        final_json_data[key] = processed_record

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_data, f, ensure_ascii=False, indent=4)
        print(f"转换成功！JSON 文件已保存至: {output_path}")
    except Exception as e:
        print(f"写入 JSON 文件时发生错误: {e}")


if __name__ == "__main__":
    INPUT_EXCEL_FILE = 'bench-key.xlsx'
    SHEET_TO_CONVERT = 'Sheet1'
    OUTPUT_JSON_FILE = 'bench_config.json'
    PRIMARY_KEY = 'bench_name'
    
    if not os.path.exists(INPUT_EXCEL_FILE):
        print(f"错误: 输入文件 '{INPUT_EXCEL_FILE}' 不存在。请确保文件与脚本在同一目录下，或提供正确路径。")
    else:
        convert_excel_to_json(
            excel_path=INPUT_EXCEL_FILE,
            sheet_name=SHEET_TO_CONVERT,
            output_path=OUTPUT_JSON_FILE,
            primary_key_col=PRIMARY_KEY
        )

