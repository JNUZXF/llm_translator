"""
BGE Reranker使用示例
功能：展示如何使用本地BGE-reranker-v2-m3模型进行文本重排序
路径：agent/utils/example_reranker_usage.py
"""

from agent_tool_bge_reranker import BGEReranker, rerank_with_local_bge

def main():
    """
    【单一职责原则】主函数演示BGE Reranker的基本用法
    """
    # 示例查询
    query = "Python深度学习框架"
    
    # 示例段落
    passages = [
        "TensorFlow是Google开发的开源机器学习框架，广泛用于深度学习应用。",
        "PyTorch是Facebook开发的深度学习框架，以其动态计算图而闻名。",
        "Keras是一个高级神经网络API，可以在TensorFlow之上运行。",
        "NumPy是Python科学计算的基础库，提供多维数组对象。",
        "Pandas是Python数据分析和处理的核心库。",
        "今天天气很好，适合外出散步。"
    ]
    
    print("="*60)
    print("BGE Reranker使用示例")
    print("="*60)
    print(f"查询: {query}")
    print(f"待排序段落数量: {len(passages)}")
    print()
    
    # 【日志】方法1：使用类的方式
    print("方法1：使用BGEReranker类")
    print("-" * 30)
    
    try:
        # 初始化reranker
        reranker = BGEReranker(local_model_path="agent/models/bge-reranker-v2-m3")
        
        # 进行重排序
        import time
        start_time = time.time()
        results = reranker.rerank_passages(query, passages, top_k=3)
        end_time = time.time()
        
        print(f"重排序完成，耗时: {end_time - start_time:.2f}秒")
        print("Top 3 相关段落：")
        for i, (passage, score, original_idx) in enumerate(results, 1):
            print(f"{i}. [分数: {score:.4f}] {passage}")
        
    except Exception as e:
        print(f"方法1执行出错: {e}")
    
    print()
    
    # 【日志】方法2：使用便捷函数
    print("方法2：使用便捷函数")
    print("-" * 30)
    
    try:
        start_time = time.time()
        results2 = rerank_with_local_bge(query, passages, top_k=3)
        end_time = time.time()
        
        print(f"重排序完成，耗时: {end_time - start_time:.2f}秒")
        print("Top 3 相关段落：")
        for i, (passage, score, original_idx) in enumerate(results2, 1):
            print(f"{i}. [分数: {score:.4f}] {passage}")
        
    except Exception as e:
        print(f"方法2执行出错: {e}")
    
    print()
    print("="*60)
    print("使用完成！")

if __name__ == "__main__":
    main() 