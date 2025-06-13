#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数集合，包含：
1. 检查和修复多标签二值化器
2. 测试模型
3. 检查项目文件
"""

import os
import sys
import torch
import joblib
import numpy as np
from gensim.models import Word2Vec
import jieba
from sklearn.preprocessing import MultiLabelBinarizer

def check_mlb(mlb_path='multi_label_binarizer.pkl'):
    """检查多标签二值化器"""
    try:
        mlb = joblib.load(mlb_path)
        print("类别:", mlb.classes_)
        print("类别数量:", len(mlb.classes_))
        return mlb
    except Exception as e:
        print(f"加载多标签二值化器失败: {e}")
        return None

def fix_mlb(input_path='multi_label_binarizer.pkl', output_path='multi_label_binarizer_fixed.pkl'):
    """修复多标签二值化器"""
    # 加载当前的MultiLabelBinarizer
    print("加载当前的MultiLabelBinarizer...")
    try:
        current_mlb = joblib.load(input_path)
        print(f"当前类别: {current_mlb.classes_}")
        print(f"当前类别数量: {len(current_mlb.classes_)}")
    except Exception as e:
        print(f"加载多标签二值化器失败: {e}")
        return False

    # 定义所有9个类别
    all_classes = np.array(['边塞诗', '田园诗', '婉约词', '豪放词', '山水诗', '送别诗', '咏史诗', '悼亡诗', '爱情诗'])
    print(f"目标类别: {all_classes}")
    print(f"目标类别数量: {len(all_classes)}")

    # 创建新的MultiLabelBinarizer并设置类别
    new_mlb = MultiLabelBinarizer()
    new_mlb.classes_ = all_classes

    # 保存新的MultiLabelBinarizer
    print(f"保存新的MultiLabelBinarizer到 {output_path}...")
    joblib.dump(new_mlb, output_path)

    print(f"修复完成。请使用 '{output_path}' 替换原来的文件。")
    return True

def test_model(test_poem="大漠孤烟直，长河落日圆。"):
    """测试模型"""
    # 检查必要的文件是否存在
    required_files = [
        'app.py',
        'keywords_data.py',
        'models/custom_dl_model.pth',
        'word2vec_model.bin',
        'multi_label_binarizer.pkl'
    ]

    print("检查必要文件...")
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"警告: 以下文件缺失: {', '.join(missing_files)}")
        return False
    else:
        print("所有必要文件都存在。")

    # 尝试加载模型
    print("\n尝试加载模型...")
    try:
        # 加载 Word2Vec 模型
        word2vec_model = Word2Vec.load("word2vec_model.bin")
        print(f"Word2Vec 模型加载成功，向量维度: {word2vec_model.vector_size}")
        
        # 加载多标签二值化器
        mlb = joblib.load('multi_label_binarizer.pkl')
        print(f"MultiLabelBinarizer 加载成功，类别: {mlb.classes_}")
        print(f"类别数量: {len(mlb.classes_)}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 定义 LSTM 模型类
        class LSTMClassifier(torch.nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMClassifier, self).__init__()
                self.hidden_size = hidden_size
                self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = torch.nn.Linear(hidden_size, output_size)
        
            def forward(self, features):
                features = features.unsqueeze(1)  # Add a sequence length dimension
                lstm_out, _ = self.lstm(features)
                output = self.fc(lstm_out[:, -1, :])  # Take the last hidden state
                return output
        
        # 加载 LSTM 模型
        num_labels = len(mlb.classes_)
        model = LSTMClassifier(input_size=word2vec_model.vector_size, hidden_size=128, output_size=num_labels)
        model.load_state_dict(torch.load('./models/custom_dl_model.pth', map_location=device))
        model = model.to(device)
        model.eval()
        print("LSTM 模型加载成功")
        
        # 测试诗句
        segmented_input = list(jieba.cut(test_poem))
        
        # 获取诗词向量
        vectors = [word2vec_model.wv[word] for word in segmented_input if word in word2vec_model.wv]
        if vectors:
            input_vec = np.mean(vectors, axis=0)
            input_vec = input_vec.reshape(1, -1)
            
            # 转换为 PyTorch 张量
            input_tensor = torch.tensor(input_vec, dtype=torch.float).to(device)
            
            # 预测
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.sigmoid(outputs)[0]
                
                # 打印结果
                print("\n测试诗句预测结果:")
                print(f"诗句: {test_poem}")
                for i, prob in enumerate(probs):
                    style_name = mlb.classes_[i]
                    print(f"{style_name}: {prob.item():.4f}")
                
                # 获取最高概率的风格
                max_prob, max_idx = torch.max(probs, dim=0)
                print(f"\n预测风格: {mlb.classes_[max_idx]} (置信度: {max_prob.item():.4f})")
                return True
        else:
            print("无法为测试诗句生成向量，可能包含词汇表外的词。")
            return False
        
    except Exception as e:
        print(f"模型加载或测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_project_files():
    """检查项目文件"""
    required_files = [
        'app.py',
        'keywords_data.py',
        'models/custom_dl_model.pth',
        'word2vec_model.bin',
        'multi_label_binarizer.pkl',
        'requirements.txt',
        'packages.txt'
    ]
    
    optional_files = [
        'train_custom_dl_model.py',
        'label_data.py',
        'data_augmentation.py',
        'create_lite_model.py',
        'utils.py'
    ]
    
    print("检查必要文件...")
    missing_required = []
    for file in required_files:
        if not os.path.exists(file):
            missing_required.append(file)
    
    if missing_required:
        print(f"警告: 以下必要文件缺失: {', '.join(missing_required)}")
    else:
        print("所有必要文件都存在。")
    
    print("\n检查可选文件...")
    missing_optional = []
    for file in optional_files:
        if not os.path.exists(file):
            missing_optional.append(file)
    
    if missing_optional:
        print(f"以下可选文件缺失: {', '.join(missing_optional)}")
    else:
        print("所有可选文件都存在。")
    
    return not missing_required

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="诗词风格分类系统工具")
    parser.add_argument("--check-mlb", action="store_true", help="检查多标签二值化器")
    parser.add_argument("--fix-mlb", action="store_true", help="修复多标签二值化器")
    parser.add_argument("--test-model", action="store_true", help="测试模型")
    parser.add_argument("--check-files", action="store_true", help="检查项目文件")
    parser.add_argument("--poem", type=str, default="大漠孤烟直，长河落日圆。", help="用于测试的诗句")
    
    args = parser.parse_args()
    
    if args.check_mlb:
        check_mlb()
    elif args.fix_mlb:
        fix_mlb()
    elif args.test_model:
        test_model(args.poem)
    elif args.check_files:
        check_project_files()
    else:
        print("请指定要执行的操作。使用 --help 查看帮助。") 