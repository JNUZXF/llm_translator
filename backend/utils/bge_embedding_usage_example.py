# 文件功能：BGE嵌入功能使用示例
# 文件路径：agent/utils/bge_embedding_usage_example.py
# 展示如何使用embedding_doubao.py中新增的BGE模型功能

"""
【扩展性】【模块化设计】BGE嵌入功能使用示例

本文件展示如何使用embedding_doubao.py中新增的本地BGE模型功能。
BGE模型相比API调用具有以下优势：
- 无需网络连接
- 无API调用限制
- 更好的隐私保护
- 更快的处理速度（批量处理时）
"""

import os
import sys

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.utils.embedding_doubao import VectorDatabase, VectorSearcher

def basic_bge_usage():
    """
    【单一职责原则】基础BGE模型使用示例
    """
    print("=== 基础BGE模型使用示例 ===")
    
    # 准备测试文本
    texts = [
        "人工智能是模拟人类智能的技术。",
        "机器学习是实现人工智能的重要方法。", 
        "深度学习是机器学习的一个分支。",
        "自然语言处理研究计算机理解人类语言。",
        "计算机视觉让机器能够理解图像内容。",
        "强化学习通过奖励机制训练智能体。"
    ]
    
    # 【模块化设计】创建向量数据库，使用BGE模型
    db = VectorDatabase(save_path="example_bge_vectors.pkl")
    
    print("使用BGE模型进行向量化...")
    db.batch_vectorize(
        texts=texts,
        model_type="bge",           # 指定使用BGE模型
        max_workers=2,              # 线程数
        bge_batch_size=4,           # BGE批处理大小
        bge_model_path=None         # 使用默认模型路径
    )
    
    # 保存向量数据库
    db.save_to_file()
    print("向量数据库已保存")
    
    # 【模块化设计】创建搜索器，使用BGE模型
    searcher = VectorSearcher(
        db_path="example_bge_vectors.pkl",
        model="bge-m3",             # 模型名称（用于日志）
        model_type="bge",           # 指定使用BGE模型
        bge_model_path=None         # 使用默认模型路径
    )
    
    # 加载数据库并执行搜索
    if searcher.load_database():
        query = "什么是深度学习"
        results = searcher.search(query, top_k=3, threshold=0.3)
        
        print(f"\n查询: {query}")
        print("搜索结果:")
        for i, (text, score) in enumerate(results, 1):
            print(f"{i}. 相似度: {score:.4f}")
            print(f"   文本: {text}")
        
    # 清理文件
    try:
        os.remove("example_bge_vectors.pkl")
        print("\n示例文件已清理")
    except:
        pass

def advanced_bge_usage():
    """
    【性能设计】高级BGE模型使用示例：自定义参数
    """
    print("\n=== 高级BGE模型使用示例 ===")
    
    # 更多测试文本
    texts = [
        "Python是一种解释型编程语言。",
        "JavaScript常用于Web前端开发。", 
        "Java是一种面向对象的编程语言。",
        "C++提供了高性能的系统编程能力。",
        "Go语言适合构建并发应用程序。",
        "Rust是一种内存安全的系统编程语言。",
        "SQL用于数据库查询和管理。",
        "HTML是网页标记语言的标准。",
        "CSS用于控制网页的样式和布局。",
        "机器学习库如TensorFlow和PyTorch很流行。"
    ]
    
    # 【性能设计】使用优化的参数设置
    db = VectorDatabase(save_path="advanced_bge_vectors.pkl")
    
    print("使用优化参数进行BGE向量化...")
    db.batch_vectorize(
        texts=texts,
        model_type="bge",           # 使用BGE模型
        max_workers=4,              # 增加线程数
        bge_batch_size=8,           # 增大批处理大小
        bge_model_path=None         # 可以指定自定义模型路径
    )
    
    db.save_to_file()
    
    # 执行多个查询测试
    searcher = VectorSearcher(
        db_path="advanced_bge_vectors.pkl",
        model="bge-m3",
        model_type="bge"
    )
    
    if searcher.load_database():
        queries = [
            "哪些是编程语言",
            "Web开发相关技术",
            "机器学习工具"
        ]
        
        for query in queries:
            print(f"\n查询: {query}")
            results = searcher.search(query, top_k=3, threshold=0.2)
            
            for i, (text, score) in enumerate(results, 1):
                print(f"  {i}. {score:.3f} | {text}")
    
    # 清理文件
    try:
        os.remove("advanced_bge_vectors.pkl")
        print("\n高级示例文件已清理")
    except:
        pass

def model_comparison_demo():
    """
    【测试策略】模型对比演示：BGE vs Doubao API
    """
    print("\n=== 模型对比演示 ===")
    
    # 检查Doubao API是否可用
    has_doubao = bool(os.environ.get("DOUBAO_API_KEY"))
    
    if not has_doubao:
        print("未设置DOUBAO_API_KEY，仅演示BGE模型")
    
    test_texts = [
        "区块链是一种分布式账本技术。",
        "加密货币基于区块链技术实现。",
        "智能合约能够自动执行协议条款。"
    ]
    
    query = "什么是区块链技术"
    
    # 【性能设计】BGE模型测试
    print("\n--- BGE模型测试 ---")
    import time
    
    start_time = time.time()
    
    db_bge = VectorDatabase(save_path="bge_compare.pkl")
    db_bge.batch_vectorize(
        texts=test_texts,
        model_type="bge",
        max_workers=2,
        bge_batch_size=3
    )
    db_bge.save_to_file()
    
    searcher_bge = VectorSearcher(
        db_path="bge_compare.pkl",
        model="bge-m3",
        model_type="bge"
    )
    
    if searcher_bge.load_database():
        results_bge = searcher_bge.search(query, top_k=2, threshold=0.0)
        bge_time = time.time() - start_time
        
        print(f"BGE处理时间: {bge_time:.2f}秒")
        print("BGE搜索结果:")
        for i, (text, score) in enumerate(results_bge, 1):
            print(f"  {i}. {score:.3f} | {text}")
    
    # 【性能设计】Doubao API测试（如果可用）
    if has_doubao:
        print("\n--- Doubao API测试 ---")
        start_time = time.time()
        
        db_doubao = VectorDatabase(save_path="doubao_compare.pkl")
        db_doubao.batch_vectorize(
            texts=test_texts,
            model_type="doubao",
            max_workers=2,
            model="doubao-embedding-text-240715"
        )
        db_doubao.save_to_file()
        
        searcher_doubao = VectorSearcher(
            db_path="doubao_compare.pkl",
            model="doubao-embedding-text-240715",
            model_type="doubao"
        )
        
        if searcher_doubao.load_database():
            results_doubao = searcher_doubao.search(query, top_k=2, threshold=0.0)
            doubao_time = time.time() - start_time
            
            print(f"Doubao处理时间: {doubao_time:.2f}秒")
            print("Doubao搜索结果:")
            for i, (text, score) in enumerate(results_doubao, 1):
                print(f"  {i}. {score:.3f} | {text}")
    
    # 清理文件
    for file in ["bge_compare.pkl", "doubao_compare.pkl"]:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass

def usage_tips():
    """
    【文档原则】使用建议和最佳实践
    """
    print("\n=== 使用建议和最佳实践 ===")
    
    tips = [
        "1. 【性能优化】大量文本处理时，优先选择BGE模型（无API限制）",
        "2. 【批处理优化】调整bge_batch_size参数，根据内存大小优化（推荐16-64）",
        "3. 【线程优化】max_workers建议设置为CPU核心数的1-2倍",
        "4. 【模型选择】对于中文文本，BGE-M3模型表现优异",
        "5. 【缓存策略】BGE模型支持结果缓存，重复文本处理更快",
        "6. 【内存管理】处理大量文本时，可以分批进行避免内存溢出",
        "7. 【路径配置】自定义bge_model_path可以使用特定版本的模型",
        "8. 【错误处理】BGE模型不可用时会自动回退到Doubao API",
        "9. 【日志监控】查看embedding_operations.log了解处理详情",
        "10. 【兼容性】新功能完全向后兼容，现有代码无需修改"
    ]
    
    for tip in tips:
        print(tip)

if __name__ == "__main__":
    print("【模块化设计】BGE嵌入功能使用示例")
    print("=" * 50)
    
    try:
        # 基础使用示例
        basic_bge_usage()
        
        # 高级使用示例
        advanced_bge_usage()
        
        # 模型对比演示
        model_comparison_demo()
        
        # 使用建议
        usage_tips()
        
        print("\n" + "=" * 50)
        print("【扩展性】所有示例演示完成！")
        print("您现在可以在自己的项目中使用BGE模型功能了。")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保agent_tool_bge_embedder.py文件存在并且依赖已安装")
    except Exception as e:
        print(f"运行错误: {e}")
        print("请检查模型文件和依赖是否正确安装") 