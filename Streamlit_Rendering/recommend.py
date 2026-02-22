
import numpy as np
import pandas as pd
import json

def _cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_recommendations_by_article(df_all, target_id, top_n=3):
    # 실제 구현 시에는 DB의 article_chunks에서 target_id와 유사한 다른 article_id를 찾는 쿼리 권장
    # 여기서는 간단히 상위 리스트 반환으로 대체 가능
    return df_all[df_all["article_id"] != target_id].head(top_n)
