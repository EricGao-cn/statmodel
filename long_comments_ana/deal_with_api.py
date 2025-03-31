import pandas as pd
import openai
import time
import json
import os
from tqdm import tqdm
import logging


# Define a function to score a comment
def score_comment(comment):
    system = '''
    你是一个专业的电影评论分析师，请根据用户输入的评论内容，从**评论的情感倾向**和**评论中对电影内容的分析**两个大方面进行评分，且内容分析从*故事叙述*, *角色表演*, *制作水准*三个方面分别打分，共打4个分数，并给出最终的综合分数。具体要求如下：

    ## 评分规则
    1. 情感倾向（sentiment_score）：
    - 0分=极端负面（人身攻击/全盘否定）
    - 1分=强烈反对或明显反讽（如"这特效真是‘国际水准’"）
    - 2分=温和批评或隐性贬低（如"电影很有勇气"）
    - 3分=绝对中立（无感情色彩的事实陈述）
    - 4分=有限支持（认可部分优点）
    - 5分=强烈支持（热情推荐/情感共鸣）

    2. 内容分析 
    - 故事叙述（story_score）：
        - 0分=彻底崩坏（逻辑混乱无法理解）
        - 1分=严重缺陷（主线模糊/硬伤多）
        - 2分=平庸叙事（完整但平淡）
        - 3分=合格水准（结构清晰但无新意）
        - 4分=优秀设计（多线叙事/伏笔精妙）
        - 5分=大师级叙事（深刻主题/强烈共鸣）
    - 角色表演（acting_score）：
        - 0分=灾难级演技（严重影响观感）
        - 1分=明显瑕疵（表情僵硬/角色扁平）
        - 2分=勉强及格（基础达标无亮点）
        - 3分=稳定发挥（自然但无突破）
        - 4分=精彩演绎（细腻有爆发力）
        - 5分=影史级表演（塑造经典角色）
    - 制作水准（production_score）：
        - 0分=粗制滥造（重大技术失误）
        - 1分=廉价感明显（五毛特效/穿帮）
        - 2分=基础达标（无失误无特色）
        - 3分=工业水准（技术合格无风格）
        - 4分=精致制作（创意视听语言）
        - 5分=艺术级呈现（开创性技术）

    ## **输出要求**
    1. 返回 JSON 格式：
    ```json
    {
        "comment": "原始评论内容",
        "sentiment_score": 0-5,
        "storytelling": {
            "score": 0-5,
            "valid": 如果评论中不包括这部分的评价则为0，否则就为1
        }
        "character_and_performance": {
            "score": 0-5,
            "valid": 如果评论中不包括这部分的评价则为0，否则就为1
        }
        "production": {
            "score": 0-5,
            "valid": 如果评论中不包括这部分的评价则为0，否则就为1
        }
        "conclusion": "对评论内容与情感的简单总结（10字以内）"
        "valid_comment": "评论是否有效，若评论与电影内容完全无关则输出0，否则输出1“
    }
    '''

    prompt = f"""
    请对以下评论进行评分：
    
    评论: {comment}
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        result = response.choices[0].message.content.strip()
        # 尝试解析JSON结果
        try:
            # 删除非JSON部分
            json_str = result
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0].strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {result}")
            return None
    except Exception as e:
        print(f"Error scoring comment: {e}")
        return None


OPENAI_API_KEY = "sk-0a7c55f0e6eb4661855c3b5b420dd108" # TODO: 把这个地方改成自己的 api key
client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com")

file_path = 'data/all_data_cleaned.xlsx'  # TODO: 把这一行改成自己的文件地址
output_file = 'data/scored_comments_all.jsonl' # TODO: 把这一行改成自己的输出文件地址
comments_data = pd.read_excel(file_path)

# comments_data = comments_data.head(4000)
comments_data = comments_data.iloc[3610:4000]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='comment_scoring.log',
    filemode='a'
)

logging.info(f"Starting comment scoring for {len(comments_data)} comments")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    pass

try:
    for idx, row in tqdm(comments_data.iterrows(), total=len(comments_data)):
        try:
            if 'content' in row and pd.notna(row['content']):
                logging.debug(f"Processing comment ID: {row.get('id', idx)}")
                result = score_comment(row['content'])
                if result:
                    if 'id' in row:
                        result['id'] = row['id']
                    
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    logging.debug(f"Successfully scored comment ID: {row.get('id', idx)}")
                else:
                    logging.warning(f"Failed to score comment ID: {row.get('id', idx)}")
                
                time.sleep(0.5)  # Rate limiting
        except Exception as e:
            logging.error(f"Error processing comment ID {row.get('id', idx)}: {str(e)}")
except Exception as e:
    logging.critical(f"Critical error in processing loop: {str(e)}")

logging.info(f"Comment scoring completed. Results saved to {output_file}")

print(f"评分完成，结果已保存到 {output_file}")
