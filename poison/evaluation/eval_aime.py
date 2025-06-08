import os
import json
import argparse
import re
from typing import List, Dict, Optional
import numpy as np

def extract_answer_from_response(response: str) -> Optional[str]:
    """
    从模型响应中提取数值答案
    AIME答案通常是0-999之间的整数
    """
    # 移除thinking标签内容
    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    
    # 寻找常见的答案模式
    patterns = [
        r'答案是\s*(\d{1,3})',
        r'答案为\s*(\d{1,3})',
        r'Therefore,?\s*the answer is\s*(\d{1,3})',
        r'So,?\s*the answer is\s*(\d{1,3})',
        r'The answer is\s*(\d{1,3})',
        r'\\boxed\{(\d{1,3})\}',
        r'答案：\s*(\d{1,3})',
        r'最终答案是\s*(\d{1,3})',
        r'最终答案为\s*(\d{1,3})',
        r'Final answer:\s*(\d{1,3})',
        r'Answer:\s*(\d{1,3})',
        r'结果是\s*(\d{1,3})',
        r'结果为\s*(\d{1,3})',
        r'所以答案是\s*(\d{1,3})',
        r'因此答案是\s*(\d{1,3})',
        r'故答案为\s*(\d{1,3})',
        r'=\s*(\d{1,3})\s*$',  # 以等号结尾的数字
        r'\b(\d{1,3})\s*$',    # 以数字结尾
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1]  # 取最后一个匹配
            try:
                num = int(answer)
                if 0 <= num <= 999:  # AIME答案范围
                    return str(num)
            except ValueError:
                continue
    
    # 如果没有找到明确答案，尝试提取所有3位以内的数字，取最后一个
    numbers = re.findall(r'\b(\d{1,3})\b', response)
    if numbers:
        for num_str in reversed(numbers):
            try:
                num = int(num_str)
                if 0 <= num <= 999:
                    return str(num)
            except ValueError:
                continue
    
    return None

def load_ground_truth(dataset_name: str) -> Dict[str, str]:
    """
    加载正确答案
    """
    ground_truth = {}
    
    if "aime24" in dataset_name.lower():
        # 从本地文件加载 AIME24 数据
        aime24_path = "/mnt/workspace/wzn/AIME/aime24.jsonl"
        
        try:
            with open(aime24_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        # 处理可能的不同字段名
                        problem = example.get("Problem", example.get("problem", ""))
                        answer = str(example.get("Answer", example.get("answer", "")))
                        if problem and answer:
                            ground_truth[problem] = answer
        except FileNotFoundError:
            print(f"错误: 找不到文件 {aime24_path}")
            return {}
        except Exception as e:
            print(f"错误: 读取文件 {aime24_path} 时出现问题: {e}")
            return {}
            
    elif "aime25" in dataset_name.lower():
        # 从本地文件加载 AIME25 数据
        aime25_path = "/mnt/workspace/wzn/AIME/aime25.jsonl"
        
        try:
            with open(aime25_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        # 处理可能的不同字段名
                        problem = example.get("problem", example.get("Problem", ""))
                        answer = str(example.get("answer", example.get("Answer", "")))
                        if problem and answer:
                            ground_truth[problem] = answer
        except FileNotFoundError:
            print(f"错误: 找不到文件 {aime25_path}")
            return {}
        except Exception as e:
            print(f"错误: 读取文件 {aime25_path} 时出现问题: {e}")
            return {}
    
    return ground_truth

def evaluate_predictions(input_path: str, dataset_name: str, verbose: bool = False) -> Dict:
    """
    评估预测结果
    """
    # 加载预测结果
    with open(input_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # 加载正确答案
    ground_truth = load_ground_truth(dataset_name)
    
    # 评估指标
    total_problems = len(predictions)
    correct_predictions = 0
    extracted_answers = 0
    evaluation_results = []
    
    for pred in predictions:
        instruction = pred['instruction']
        model_output = pred.get('output', pred.get('raw_output', ''))
        
        # 提取模型答案
        predicted_answer = extract_answer_from_response(model_output)
        
        # 获取正确答案
        true_answer = ground_truth.get(instruction)
        
        # 评估结果
        result = {
            'instruction': instruction,
            'predicted_answer': predicted_answer,
            'true_answer': true_answer,
            'model_output': model_output,
            'correct': False,
            'answer_extracted': predicted_answer is not None
        }
        
        if predicted_answer is not None:
            extracted_answers += 1
            if true_answer is not None and predicted_answer == true_answer:
                correct_predictions += 1
                result['correct'] = True
        
        evaluation_results.append(result)
        
        if verbose:
            status = "✓" if result['correct'] else "✗"
            extract_status = "📝" if result['answer_extracted'] else "❌"
            print(f"{status} {extract_status} Pred: {predicted_answer} | True: {true_answer}")
            if len(instruction) > 100:
                print(f"   Problem: {instruction[:100]}...")
            else:
                print(f"   Problem: {instruction}")
            print()
    
    # 计算指标
    accuracy = correct_predictions / total_problems if total_problems > 0 else 0
    extraction_rate = extracted_answers / total_problems if total_problems > 0 else 0
    accuracy_on_extracted = correct_predictions / extracted_answers if extracted_answers > 0 else 0
    
    metrics = {
        'total_problems': total_problems,
        'correct_predictions': correct_predictions,
        'extracted_answers': extracted_answers,
        'accuracy': accuracy,
        'extraction_rate': extraction_rate,
        'accuracy_on_extracted': accuracy_on_extracted,
        'evaluation_results': evaluation_results
    }
    
    return metrics

def print_evaluation_summary(metrics: Dict, dataset_name: str):
    """
    打印评估摘要
    """
    print("=" * 60)
    print(f"AIME 评估结果总结 - {dataset_name}")
    print("=" * 60)
    print(f"总题目数量: {metrics['total_problems']}")
    print(f"成功提取答案的题目数量: {metrics['extracted_answers']}")
    print(f"正确预测的题目数量: {metrics['correct_predictions']}")
    print()
    print(f"整体准确率: {metrics['accuracy']:.2%} ({metrics['correct_predictions']}/{metrics['total_problems']})")
    print(f"答案提取率: {metrics['extraction_rate']:.2%} ({metrics['extracted_answers']}/{metrics['total_problems']})")
    print(f"提取答案中的准确率: {metrics['accuracy_on_extracted']:.2%} ({metrics['correct_predictions']}/{metrics['extracted_answers']})")
    print("=" * 60)

def analyze_errors(metrics: Dict, top_k: int = 5):
    """
    分析错误案例
    """
    print(f"\n错误案例分析 (显示前 {top_k} 个):")
    print("-" * 60)
    
    error_cases = [r for r in metrics['evaluation_results'] if not r['correct']]
    
    for i, case in enumerate(error_cases[:top_k]):
        print(f"\n错误案例 {i+1}:")
        print(f"问题: {case['instruction'][:150]}...")
        print(f"预测答案: {case['predicted_answer']}")
        print(f"正确答案: {case['true_answer']}")
        print(f"答案是否提取成功: {'是' if case['answer_extracted'] else '否'}")
        
        if case['model_output']:
            print(f"模型输出: {case['model_output'][:200]}...")
        print("-" * 40)

def save_detailed_results(metrics: Dict, output_file: str):
    """
    保存详细评估结果
    """
    detailed_results = {
        'summary': {
            'total_problems': metrics['total_problems'],
            'correct_predictions': metrics['correct_predictions'],
            'extracted_answers': metrics['extracted_answers'],
            'accuracy': metrics['accuracy'],
            'extraction_rate': metrics['extraction_rate'],
            'accuracy_on_extracted': metrics['accuracy_on_extracted']
        },
        'detailed_results': metrics['evaluation_results']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细评估结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="评估AIME数据集上的模型预测结果")
    parser.add_argument("--input_path", required=True, help="pred_adaptive.py生成的预测结果文件")
    parser.add_argument("--dataset_name", choices=['aime24', 'aime25'], default='aime25',
                       help="数据集名称")
    parser.add_argument("--output_file", default="", help="详细评估结果输出文件")
    parser.add_argument("--verbose", action="store_true", help="显示详细的逐题评估结果")
    parser.add_argument("--analyze_errors", type=int, default=5, help="分析错误案例的数量")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"错误: 预测文件 {args.input_path} 不存在")
        return
    
    print(f"正在评估文件: {args.input_path}")
    print(f"使用数据集: {args.dataset_name}")
    print()
    
    # 执行评估
    metrics = evaluate_predictions(args.input_path, args.dataset_name, args.verbose)
    
    # 打印总结
    print_evaluation_summary(metrics, args.dataset_name)
    
    # 分析错误案例
    if args.analyze_errors > 0:
        analyze_errors(metrics, args.analyze_errors)
    
    # 保存详细结果
    if args.output_file:
        save_detailed_results(metrics, args.output_file)
    elif args.input_path.endswith('.json'):
        default_output = args.input_path.replace('.json', '_evaluation.json')
        save_detailed_results(metrics, default_output)

if __name__ == "__main__":
    main() 