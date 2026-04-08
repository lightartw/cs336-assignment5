import json

def count_evaluation_metrics(file_path):
    cat_1 = 0  # (1) format 1, answer 1
    cat_2 = 0  # (2) format 1, answer 0
    cat_3 = 0  # (3) format 0, answer 0
    total = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for entry in data:
            total += 1
            metrics = entry.get("metrics", {})
            f_reward = metrics.get("format_reward", 0.0)
            a_reward = metrics.get("answer_reward", 0.0)

            if f_reward == 1.0 and a_reward == 1.0:
                cat_1 += 1
            elif f_reward == 1.0 and a_reward == 0.0:
                cat_2 += 1
            elif f_reward == 0.0 and a_reward == 0.0:
                cat_3 += 1
        
        print(f"--- 统计结果 (总计: {total}) ---")
        print(f"(1) Both Reward 1 (格式正确且答案正确): {cat_1}")
        print(f"(2) Format 1, Answer 0 (格式正确但答案错误): {cat_2}")
        print(f"(3) Both Reward 0 (格式错误): {cat_3}")
        
    except Exception as e:
        print(f"读取文件出错: {e}")

count_evaluation_metrics('../results/evaluation_results.json')