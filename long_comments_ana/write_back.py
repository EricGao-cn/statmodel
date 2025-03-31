import json
import pandas as pd

def write_scores_to_data():
    # Load the scored comments
    scored_comments = []
    with open('data/scored_comments_all.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            scored_comments.append(json.loads(line.strip()))
    
    # Load the original data
    try:
        data = pd.read_excel('data/all_data_cleaned.xlsx')
    except:
        data = pd.read_excel('../data/all_data_cleaned.xlsx')
    
    # Create a dictionary with comments as keys for faster lookup
    scored_dict = {item['comment']: item for item in scored_comments}
    
    # Create new columns for the scores
    data['sentiment_score'] = None
    data['storytelling_score'] = None
    data['storytelling_valid'] = None
    data['character_performance_score'] = None
    data['character_performance_valid'] = None
    data['production_score'] = None
    data['production_valid'] = None
    data['conclusion'] = None
    data['valid_comment'] = None
    
    # Get the column name that contains comments
    comment_column = "content"
    
    # Map scores back to original data
    for idx, row in data.iterrows():
        comment = row[comment_column]
        if comment in scored_dict:
            score_data = scored_dict[comment]
            data.at[idx, 'sentiment_score'] = score_data['sentiment_score']
            data.at[idx, 'storytelling_score'] = score_data['storytelling']['score']
            data.at[idx, 'storytelling_valid'] = score_data['storytelling']['valid']
            data.at[idx, 'character_performance_score'] = score_data['character_and_performance']['score'] 
            data.at[idx, 'character_performance_valid'] = score_data['character_and_performance']['valid']
            data.at[idx, 'production_score'] = score_data['production']['score']
            data.at[idx, 'production_valid'] = score_data['production']['valid']
            data.at[idx, 'conclusion'] = score_data['conclusion']
            data.at[idx, 'valid_comment'] = score_data['valid_comment']
    
    # Save the updated data
    # data.to_csv('data/all_data_cleaned_with_scores.csv', index=False)
    data.to_excel('data/all_data_cleaned_with_scores.xlsx', index=False)
    print(f"Successfully wrote scores to all_data_cleaned_with_scores.csv")

if __name__ == "__main__":
    write_scores_to_data()