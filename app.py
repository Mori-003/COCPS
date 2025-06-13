import streamlit as st
import torch
import numpy as np
import jieba
from gensim.models import Word2Vec
import joblib
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from train_custom_dl_model import HierarchicalClassifier, GENRE_LABELS, STYLE_LABELS

# 设置页面配置
st.set_page_config(page_title="诗词风格诊断系统", layout="wide")

# 设置标题
st.title("诗词风格诊断系统")

# 检查必要的模型文件是否存在
required_files = [
    "word2vec_model.bin",
    "genre_label_binarizer.pkl",
    "style_label_binarizer_诗.pkl",
    "style_label_binarizer_词.pkl",
    "models/hierarchical_model.pth"
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"缺少以下必要文件: {', '.join(missing_files)}")
    st.info("请先运行 train_custom_dl_model.py 训练模型。")
    st.stop()

# 加载模型和必要的组件
@st.cache_resource
def load_models():
    try:
        # 加载Word2Vec模型
        word2vec_model = Word2Vec.load("word2vec_model.bin")
        
        # 加载标签二值化器
        genre_mlb = joblib.load("genre_label_binarizer.pkl")
        style_mlbs = {
            '诗': joblib.load("style_label_binarizer_诗.pkl"),
            '词': joblib.load("style_label_binarizer_词.pkl")
        }
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        input_size = word2vec_model.vector_size
        hidden_size = 256
        genre_output_size = len(GENRE_LABELS)
        style_output_sizes = {genre: len(styles) for genre, styles in STYLE_LABELS.items()}
        
        model = HierarchicalClassifier(
            input_size, hidden_size, genre_output_size, style_output_sizes
        )
        model.load_state_dict(torch.load("models/hierarchical_model.pth", map_location=device))
        model.to(device)
        model.eval()
        
        return word2vec_model, genre_mlb, style_mlbs, model, device
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}")
        return None, None, None, None, None

word2vec_model, genre_mlb, style_mlbs, model, device = load_models()

# 定义意象词典 - 按风格分类的关键字
imagery_dict = {
    '边塞诗': ['烽火', '长城', '戍边', '胡马', '征人', '战场', '关山', '大漠', '雁', '风沙', '边塞'],
    '山水诗': ['山', '水', '云', '溪', '江', '林', '泉', '石', '松', '峰', '月', '雾', '湖'],
    '田园诗': ['农', '牧', '桑', '田', '园', '村', '草', '禾', '鸡', '犬', '桃', '柳', '杏', '耕', '锄', '父', '童', '野', '年丰', '岁', '农家', '田家', '村家', 
              '归园', '田居', '隐居', '采菊', '东篱', '南山', '饮酒', '种豆', '种瓜', '种菜', '耘田', '牛羊', '鸡犬', '桑麻', '稻粱', '柴门', '茅舍', '茅屋', '茅庐', '篱笆'],
    '咏物诗': ['花', '鸟', '鱼', '虫', '竹', '梅', '兰', '菊', '松', '枫', '雪', '雨', '风'],
    '咏史诗': ['古', '史', '帝', '王', '将', '臣', '战', '兴', '亡', '传', '汉', '唐', '宋'],
    '送别诗': ['别', '送', '归', '离', '望', '思', '远', '亭', '楼', '酒', '泪', '路', '程'],
    '豪放词': ['豪', '壮', '气', '天', '山', '河', '国', '英', '雄', '笑', '战', '酒', '歌'],
    '婉约词': ['情', '愁', '恨', '别', '怨', '泪', '梦', '闺', '花', '月', '柔', '香', '思']
}

# 风格特征向量 - 用于相似度计算
style_vectors = {}
for style, keywords in imagery_dict.items():
    style_vec = np.zeros(word2vec_model.vector_size)
    count = 0
    for word in keywords:
        if word in word2vec_model.wv:
            style_vec += word2vec_model.wv[word]
            count += 1
    if count > 0:
        style_vec /= count
    style_vectors[style] = style_vec

# 定义预测函数
def predict_style(text, thresholds=None):
    # 预处理文本 - 移除多余空格和标点
    text = re.sub(r'\s+', '', text)  # 移除所有空白字符
    text = re.sub(r'[^\u4e00-\u9fa5，。！？、]', '', text)  # 仅保留中文和基本标点
    
    # 分词
    tokens = list(jieba.cut(text))
    
    # 将分词后的文本转换为Word2Vec向量序列
    MAX_LEN = 128  # 与训练时保持一致
    sequence = []
    word_vectors = []  # 保存每个词的向量，用于意象分析
    
    for token in tokens:
        if token in word2vec_model.wv:
            vec = word2vec_model.wv[token]
            sequence.append(vec)
            word_vectors.append((token, vec))
        else:
            # 对于未知词，使用零向量
            vec = np.zeros(word2vec_model.vector_size)
            sequence.append(vec)
            word_vectors.append((token, vec))
    
    if not sequence:
        return "无法识别", {}, {}, [], {}
    
    # 限制最大长度
    if len(sequence) > MAX_LEN:
        sequence = sequence[:MAX_LEN]
        word_vectors = word_vectors[:MAX_LEN]
    
    # 填充序列到最大长度
    padding_length = MAX_LEN - len(sequence)
    if padding_length > 0:
        sequence.extend([np.zeros(word2vec_model.vector_size)] * padding_length)
    
    # 转换为tensor并添加batch维度
    features = torch.tensor(np.array(sequence), dtype=torch.float).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        genre_logits, style_logits = model(features)
        
        # 获取体裁预测（诗/词）
        genre_probs = torch.softmax(genre_logits, dim=1).cpu().numpy()[0]
        
        # 基于规则的体裁分类增强
        
        # 1. 基本长度特征
        if len(tokens) < 40:  # 较短的文本倾向于是诗
            genre_probs[0] *= 1.3  # 大幅增加"诗"的权重
        elif len(tokens) > 60:  # 较长的文本倾向于是词
            genre_probs[1] *= 1.2  # 增加"词"的权重
            
        # 2. 句式结构分析
        sentences = [s for s in re.split(r'[，。！？、]', text) if s.strip()]
        sentence_lengths = [len(s) for s in sentences if s]
        
        # 检查是否有典型的诗句式规律（五言、七言或近似）
        if sentence_lengths:
            # 计算长度的平均值和方差
            avg_len = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            
            # 如果句子长度比较一致且接近5或7
            if variance < 2.0 and (4 <= avg_len <= 6 or 6 <= avg_len <= 8):
                genre_probs[0] *= 1.4  # 大幅增加"诗"的权重
                
            # 如果句子长度相差很大，可能是词
            elif variance > 4.0:
                genre_probs[1] *= 1.25  # 增加"词"的权重
                
            # 检查对偶句特征 - 诗歌常见
            if len(sentences) >= 4:
                pairs_count = 0
                for i in range(0, len(sentences)-1, 2):
                    if i+1 < len(sentences) and abs(len(sentences[i]) - len(sentences[i+1])) <= 1:  # 相邻两句长度接近
                        pairs_count += 1
                
                # 有多组对偶句
                if pairs_count >= 2:
                    genre_probs[0] *= 1.2  # 增加"诗"的权重
                
        # 3. 关键词特征检测
        # 诗歌通用关键词
        poem_keywords = ['山', '水', '月', '风', '花', '云', '雪', '天', '日', '夜', '春', '秋', '沙', '雨', '梦', '空']
        poem_keyword_count = sum(keyword in text for keyword in poem_keywords)
        if poem_keyword_count >= 2:
            genre_probs[0] *= (1.0 + 0.05 * min(poem_keyword_count, 5))  # 随着关键词增加提高权重
            
        # 田园诗关键词 - 增强识别
        farm_keywords = ['田', '农', '耕', '桑', '锄', '牧', '童', '野', '父', '禾', '年丰', '牛', '犁', '稻', 
                        '归园', '田居', '隐居', '采菊', '东篱', '南山', '种豆', '桑麻', '柴门', '茅舍']
        farm_keyword_count = sum(keyword in text for keyword in farm_keywords)
        if farm_keyword_count >= 1:
            genre_probs[0] *= (1.2 + 0.1 * min(farm_keyword_count, 3))  # 随着关键词增加提高权重
            
        # 边塞诗关键词
        frontier_keywords = ['塞', '边', '关', '征', '战', '胡', '马', '刀', '戍', '边关', '烽火']
        frontier_keyword_count = sum(keyword in text for keyword in frontier_keywords)
        if frontier_keyword_count >= 1:
            genre_probs[0] *= (1.2 + 0.1 * min(frontier_keyword_count, 3))
            
        # 词牌名检测 - 如果包含词牌名，很可能是词
        cipai_names = ['满江红', '水调歌头', '念奴娇', '浣溪沙', '醉花阴', '清平乐', '菩萨蛮', '蝶恋花', '临江仙']
        if any(cipai in text[:10] for cipai in cipai_names):  # 在文本开头检测词牌名
            genre_probs[1] *= 2.0  # 大幅增加"词"的权重
            
        # 4. 意象分布分析
        # 获取前10个最强的词向量
        top_vectors = sorted(word_vectors, key=lambda x: np.linalg.norm(x[1]), reverse=True)[:10]
        
        # 计算向量的平均余弦相似度 - 诗的意象相似度通常更高
        if len(top_vectors) >= 3:
            similarity_sum = 0
            pair_count = 0
            for i in range(len(top_vectors)):
                for j in range(i+1, len(top_vectors)):
                    sim = cosine_similarity([top_vectors[i][1]], [top_vectors[j][1]])[0][0]
                    similarity_sum += sim
                    pair_count += 1
            
            avg_similarity = similarity_sum / pair_count if pair_count > 0 else 0
            
            # 意象一致性高的可能是诗
            if avg_similarity > 0.5:
                genre_probs[0] *= 1.15
            # 意象分散的可能是词
            elif avg_similarity < 0.3:
                genre_probs[1] *= 1.1
        
        # 5. 特定诗词风格的关键句式特征
        # 田园诗特征 - 常见"归园田居"、"饮酒"等主题
        if "归" in text and ("园" in text or "田" in text or "隐" in text):
            # 增加田园诗的权重
            genre_probs[0] *= 1.3
            
        # 检查是否包含典型的田园诗标题或主题
        if any(phrase in text for phrase in ["归园田居", "饮酒", "采菊", "东篱", "南山", "种豆", "桑麻", "柴门"]):
            genre_probs[0] *= 1.4  # 大幅增加"诗"的权重
                
        # 重新归一化概率
        genre_probs = genre_probs / np.sum(genre_probs)
        
        genre_idx = np.argmax(genre_probs)
        genre = GENRE_LABELS[genre_idx]
        
        # 获取风格标签
        style_labels = STYLE_LABELS[genre]
        
        # 获取风格预测
        style_logit = style_logits[genre][0]
        style_probs = torch.sigmoid(style_logit).cpu().numpy()
        
        # 风格分类增强 - 针对特定风格的额外调整
        if genre == '诗':
            # 检测田园诗特征
            farm_theme_keywords = ['田', '农', '耕', '桑', '锄', '牧', '童', '野', '父', '禾', '归园', '田居', '隐居']
            farm_theme_count = sum(keyword in text for keyword in farm_theme_keywords)
            
            # 如果包含多个田园诗关键词，增强田园诗的概率
            if farm_theme_count >= 2:
                farm_idx = style_labels.index('田园诗') if '田园诗' in style_labels else -1
                if farm_idx >= 0:
                    style_probs[farm_idx] *= (1.2 + 0.1 * min(farm_theme_count, 5))
            
            # 检测边塞诗特征
            frontier_theme_keywords = ['塞', '边', '关', '征', '战', '胡', '马', '戍']
            frontier_theme_count = sum(keyword in text for keyword in frontier_theme_keywords)
            
            if frontier_theme_count >= 2:
                frontier_idx = style_labels.index('边塞诗') if '边塞诗' in style_labels else -1
                if frontier_idx >= 0:
                    style_probs[frontier_idx] *= (1.2 + 0.1 * min(frontier_theme_count, 5))
        
        # 应用自定义阈值（如果提供）
        if thresholds and genre in thresholds:
            style_thresholds = thresholds[genre]
            style_preds = np.zeros_like(style_probs, dtype=int)
            for i, style_name in enumerate(style_labels):
                if style_name in style_thresholds:
                    style_preds[i] = int(style_probs[i] > style_thresholds[style_name])
                else:
                    # 对于没有特定阈值的风格，使用默认阈值
                    style_preds[i] = int(style_probs[i] > 0.4)
                    
            # 应用阈值调整后的风格概率
            # 如果某个风格的概率低于阈值，将其概率降低
            for i, style_name in enumerate(style_labels):
                if style_name in style_thresholds:
                    if style_probs[i] <= style_thresholds[style_name]:
                        style_probs[i] *= 0.8  # 降低不符合阈值的风格概率
        else:
            # 默认阈值
            default_thresh = 0.4
            style_preds = (style_probs > default_thresh).astype(int)
            
            # 应用默认阈值调整
            for i in range(len(style_probs)):
                if style_probs[i] <= default_thresh:
                    style_probs[i] *= 0.8  # 降低不符合默认阈值的风格概率
        
        # 创建风格-概率字典
        style_prob_dict = {style_labels[i]: float(style_probs[i]) for i in range(len(style_labels))}
        genre_prob_dict = {GENRE_LABELS[i]: float(genre_probs[i]) for i in range(len(GENRE_LABELS))}
        
        # 计算意象贡献
        imagery_contributions = analyze_imagery(word_vectors, genre)
        
        # 计算风格纯净度
        purity_scores = calculate_style_purity(word_vectors, genre)
        
        # 获取最可能的风格 - 考虑阈值后的概率
        max_style_idx = np.argmax(style_probs)
        max_style = style_labels[max_style_idx]
        
        # 如果最高概率风格低于其阈值，标记为"混合风格"
        if thresholds and genre in thresholds:
            max_style_threshold = thresholds[genre].get(max_style, 0.4)
            if style_probs[max_style_idx] < max_style_threshold:
                # 查找第二高概率的风格
                second_probs = style_probs.copy()
                second_probs[max_style_idx] = 0
                second_idx = np.argmax(second_probs)
                second_style = style_labels[second_idx]
                
                # 如果两种风格概率接近，标记为混合风格
                if second_probs[second_idx] > 0 and style_probs[max_style_idx] / second_probs[second_idx] < 1.3:
                    max_style = f"{max_style}/混合{second_style}"
        
    return f"{genre} - {max_style}", genre_prob_dict, style_prob_dict, imagery_contributions, purity_scores

# 分析词语对各风格的贡献度
def analyze_imagery(word_vectors, genre):
    if not word_vectors:
        return []
    
    result = []
    style_names = STYLE_LABELS[genre]
    
    for token, vec in word_vectors:
        if token in ['的', '了', '在', '是', '和', '与', '或', '而', '又']:  # 忽略常见虚词
            continue
            
        # 计算词与各风格的相似度
        similarities = {}
        for style in style_names:
            if style in style_vectors:
                sim = cosine_similarity([vec], [style_vectors[style]])[0][0]
                similarities[style] = float(sim)
        
        # 仅保留相似度较高的风格（去除噪音）
        max_sim = max(similarities.values()) if similarities else 0
        if max_sim > 0.1:  # 仅保留有一定相似度的词
            result.append({
                "word": token,
                "similarities": similarities,
                "max_style": max(similarities, key=similarities.get) if similarities else None,
                "max_similarity": max_sim
            })
    
    # 按最大相似度排序
    result.sort(key=lambda x: x["max_similarity"], reverse=True)
    return result[:10]  # 返回前10个最有贡献的词

# 计算风格纯净度
def calculate_style_purity(word_vectors, genre):
    if not word_vectors:
        return {}
    
    style_names = STYLE_LABELS[genre]
    purity_scores = {style: 0.0 for style in style_names}
    
    # 计算文本的平均向量
    text_vector = np.mean([vec for _, vec in word_vectors], axis=0)
    
    # 计算文本向量与各风格向量的相似度
    for style in style_names:
        if style in style_vectors:
            sim = cosine_similarity([text_vector], [style_vectors[style]])[0][0]
            purity_scores[style] = float(sim)
    
    return purity_scores

# 生成雷达图
def plot_radar_chart(purity_scores):
    categories = list(purity_scores.keys())
    values = list(purity_scores.values())
    
    # 确保值域在0-1之间
    values = [max(0, min(v, 1)) for v in values]
    
    # 闭合雷达图
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='风格纯净度',
        line_color='rgb(0, 102, 204)',
        fillcolor='rgba(0, 102, 204, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.9, 1]  # 将范围从0-1调整为0.9-1，以便更好地显示细微差异
            )
        ),
        showlegend=False,
        title="风格纯净度雷达图"
    )
    
    return fig

# 分析诗词结构特征
def analyze_poem_structure(text):
    """分析诗词的句式结构和韵律特征"""
    # 预处理文本
    text = re.sub(r'\s+', '', text)  # 移除空白字符
    
    # 分割句子
    sentences = [s for s in re.split(r'[，。！？、]', text) if s.strip()]
    sentence_lengths = [len(s) for s in sentences if s]
    
    if not sentence_lengths:
        return {
            "structure": "无法识别",
            "pattern": "无法识别",
            "regularity": 0.0
        }
    
    # 分析句式特征
    avg_len = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
    
    # 判断诗词句式
    if variance < 1.0:  # 句子长度非常规整
        if 4.5 <= avg_len <= 5.5:
            pattern = "五言诗"
        elif 6.5 <= avg_len <= 7.5:
            pattern = "七言诗"
        else:
            pattern = f"{round(avg_len)}言诗"
    else:
        if len(sentences) >= 8:  # 长度较长
            if any(len(s) > 10 for s in sentences):
                pattern = "长短句(词)"
            else:
                pattern = "不规则诗"
        else:
            pattern = "不规则短句"
    
    # 分析整体结构
    if len(sentences) == 4:
        structure = "四句体(绝句)"
    elif len(sentences) == 8:
        structure = "八句体(律诗)"
    elif len(sentences) == 2:
        structure = "二句体(对句)"
    elif len(sentences) > 10:
        structure = "长篇/词"
    else:
        structure = f"{len(sentences)}句体"
    
    # 计算规整度 - 句式规则性的度量
    # 规整度越高，说明句式越规则
    length_set = set(sentence_lengths)
    if len(length_set) == 1:  # 所有句子长度相同
        regularity = 1.0
    else:
        # 方差越小，规整度越高
        regularity = max(0, 1.0 - (variance / 10))
    
    # 对偶性分析
    pairs = []
    if len(sentences) >= 4:
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                # 检查是否对偶 - 长度相近且有共同字
                is_paired = abs(len(sentences[i]) - len(sentences[i+1])) <= 1
                pairs.append(is_paired)
    
    duilian_rate = sum(pairs) / len(pairs) if pairs else 0
    
    return {
        "structure": structure,
        "pattern": pattern,
        "regularity": round(regularity, 2),
        "duilian_rate": round(duilian_rate, 2),
        "avg_length": round(avg_len, 2),
        "variance": round(variance, 2)
    }

# 每个标签页中展示结构分析结果
def display_structure_analysis(text):
    """在界面展示结构分析结果"""
    structure_info = analyze_poem_structure(text)
    
    # 创建表格展示
    st.subheader("句式结构分析")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("整体结构", structure_info["structure"])
        st.metric("句式类型", structure_info["pattern"])
    
    with col2:
        st.metric("规整度", f"{structure_info['regularity'] * 100:.1f}%")
        st.metric("对偶率", f"{structure_info['duilian_rate'] * 100:.1f}%")
    
    # 添加解释
    pattern_desc = ""
    if "五言" in structure_info["pattern"]:
        pattern_desc = "五言诗是每句五个字的诗歌，风格简洁明快。"
    elif "七言" in structure_info["pattern"]:
        pattern_desc = "七言诗是每句七个字的诗歌，表达更为丰富。"
    elif "长短句" in structure_info["pattern"]:
        pattern_desc = "长短句是典型的词牌结构，句式长短不一。"
        
    st.write(pattern_desc)
    
    return structure_info

# 创建主界面
st.write("请输入您想要诊断的诗词:")

# 添加示例诗词下拉框
example_poems = {
    "选择示例诗词...": "",
    "边塞诗 - 王维《使至塞上》": "单车欲问边，属国过居延。征蓬出汉塞，归雁入胡天。大漠孤烟直，长河落日圆。萧关逢候骑，都护在燕然。",
    "山水诗 - 李白《望庐山瀑布》": "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "田园诗 - 陶渊明《归园田居·其一》": "少无适俗韵，性本爱丘山。误落尘网中，一去三十年。羁鸟恋旧林，池鱼思故渊。开荒南野际，守拙归园田。",
    "田园诗 - 陶渊明《饮酒·其五》": "结庐在人境，而无车马喧。问君何能尔？心远地自偏。采菊东篱下，悠然见南山。山气日夕佳，飞鸟相与还。此中有真意，欲辨已忘言。",
    "田园诗 - 孟浩然《过故人庄》": "故人具鸡黍，邀我至田家。绿树村边合，青山郭外斜。开轩面场圃，把酒话桑麻。待到重阳日，还来就菊花。",
    "豪放词 - 苏轼《念奴娇·赤壁怀古》": "大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。乱石穿空，惊涛拍岸，卷起千堆雪。江山如画，一时多少豪杰。遥想公瑾当年，小乔初嫁了，雄姿英发。羽扇纶巾，谈笑间，樯橹灰飞烟灭。故国神游，多情应笑我，早生华发。人生如梦，一尊还酹江月。",
    "婉约词 - 李清照《如梦令·昨夜雨疏风骤》": "昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧。知否，知否？应是绿肥红瘦。",
    "易误判诗 - 佚名《昨夜斗回北》": "昨夜斗回北，今朝岁起东。我年已强壮，无禄尚忧农。桑野就耕父，荷锄随牧童。田家占气候，共说此年丰。"
}

selected_example = st.selectbox("或选择一个示例诗词:", options=list(example_poems.keys()))

# 创建标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs(["风格诊断", "意象分析", "纯净度雷达图", "诗词对比", "使用指南"])

with tab1:
    # 第一个标签页 - 基本诊断功能
    user_input = st.text_area("诗词内容:", height=150, key="input_text1", value=example_poems[selected_example])
    
    # 添加提示信息
    st.info("提示: 输入完整的诗词内容，包括标点符号，可以提高诊断的准确性。")
    
    # 交互式阈值调整
    with st.expander("高级选项: 阈值调整"):
        st.write("调整不同风格的识别阈值（值越低，越容易被识别为该风格）")
        col1, col2 = st.columns(2)
        
        thresholds = {}
        with col1:
            st.subheader("诗的风格阈值")
            for style in STYLE_LABELS['诗']:
                thresholds.setdefault('诗', {})
                thresholds['诗'][style] = st.slider(f"{style} 阈值", 0.1, 0.9, 0.4, 0.05)
        
        with col2:
            st.subheader("词的风格阈值")
            for style in STYLE_LABELS['词']:
                thresholds.setdefault('词', {})
                thresholds['词'][style] = st.slider(f"{style} 阈值", 0.1, 0.9, 0.4, 0.05)
    
    if st.button("诊断风格", key="diagnose_btn1"):
        if not user_input:
            st.error("请输入有效的中文诗词文本。")
        else:
            # 清理文本
            cleaned_text = re.sub(r'[^\u4e00-\u9fa5，。！？、]', '', user_input)
            
            with st.spinner("正在诊断..."):
                style_result, genre_probs, style_probs, imagery_contributions, purity_scores = predict_style(cleaned_text, thresholds)
                
                # 显示结果
                st.success(f"诊断结果: {style_result}")
                
                # 显示句式结构分析
                display_structure_analysis(cleaned_text)
                
                # 显示体裁分类结果
                st.subheader("体裁分类结果")
                st.bar_chart(genre_probs)
                
                # 显示风格分类结果
                st.subheader("风格分类结果")
                
                # 按概率大小排序
                sorted_probs = sorted(style_probs.items(), key=lambda x: x[1], reverse=True)
                
                # 使用条形图显示
                chart_data = {label: value for label, value in sorted_probs}
                st.bar_chart(chart_data)
                
                # 显示详细概率
                for label, prob in sorted_probs:
                    st.text(f"{label}: {prob:.4f}")

with tab2:
    # 第二个标签页 - 意象贡献分析
    user_input = st.text_area("诗词内容:", height=150, key="input_text2", value=example_poems[selected_example])
    
    # 解释说明
    st.info("意象分析可以帮助您理解每个关键词对诗词风格的贡献程度。")
    
    # 交互式阈值调整
    with st.expander("高级选项: 阈值调整"):
        st.write("调整不同风格的识别阈值（值越低，越容易被识别为该风格）")
        col1, col2 = st.columns(2)
        
        thresholds2 = {}
        with col1:
            st.subheader("诗的风格阈值")
            for style in STYLE_LABELS['诗']:
                thresholds2.setdefault('诗', {})
                thresholds2['诗'][style] = st.slider(f"{style} 阈值", 0.1, 0.9, 0.4, 0.05, key=f"tab2_{style}")
        
        with col2:
            st.subheader("词的风格阈值")
            for style in STYLE_LABELS['词']:
                thresholds2.setdefault('词', {})
                thresholds2['词'][style] = st.slider(f"{style} 阈值", 0.1, 0.9, 0.4, 0.05, key=f"tab2_{style}")
    
    if st.button("分析意象", key="analyze_btn"):
        if not user_input:
            st.error("请输入有效的中文诗词文本。")
        else:
            # 清理文本
            cleaned_text = re.sub(r'[^\u4e00-\u9fa5，。！？、]', '', user_input)
            
            with st.spinner("正在分析..."):
                style_result, genre_probs, style_probs, imagery_contributions, purity_scores = predict_style(cleaned_text, thresholds2)
                
                # 显示结果
                st.success(f"诊断结果: {style_result}")
                
                # 显示意象贡献分析
                st.subheader("意象贡献分析")
                if imagery_contributions:
                    # 创建数据表格
                    data = []
                    for item in imagery_contributions:
                        word = item["word"]
                        max_style = item["max_style"]
                        max_sim = item["max_similarity"]
                        data.append({
                            "词语": word,
                            "最匹配风格": max_style,
                            "匹配度": f"{max_sim:.4f}"
                        })
                    
                    df = pd.DataFrame(data)
                    st.table(df)
                    
                    # 可视化前5个词的所有风格贡献
                    if len(imagery_contributions) >= 3:
                        st.subheader("关键词语对各风格的贡献度")
                        
                        # 获取前3个词语
                        top_words = imagery_contributions[:3]
                        
                        # 为每个词语创建条形图
                        for item in top_words:
                            word = item["word"]
                            similarities = item["similarities"]
                            
                            st.write(f"**{word}** 对各风格的贡献度:")
                            st.bar_chart(similarities)
                else:
                    st.info("未找到显著的意象贡献。")

with tab3:
    # 第三个标签页 - 风格纯净度雷达图
    user_input = st.text_area("诗词内容:", height=150, key="input_text3", value=example_poems[selected_example])
    
    # 解释说明
    st.info("风格纯净度展示了诗词在各个风格上的特征强度，雷达图可直观展示风格的多样性。")
    
    # 交互式阈值调整
    with st.expander("高级选项: 阈值调整"):
        st.write("调整不同风格的识别阈值（值越低，越容易被识别为该风格）")
        col1, col2 = st.columns(2)
        
        thresholds3 = {}
        with col1:
            st.subheader("诗的风格阈值")
            for style in STYLE_LABELS['诗']:
                thresholds3.setdefault('诗', {})
                thresholds3['诗'][style] = st.slider(f"{style} 阈值", 0.1, 0.9, 0.4, 0.05, key=f"tab3_{style}")
        
        with col2:
            st.subheader("词的风格阈值")
            for style in STYLE_LABELS['词']:
                thresholds3.setdefault('词', {})
                thresholds3['词'][style] = st.slider(f"{style} 阈值", 0.1, 0.9, 0.4, 0.05, key=f"tab3_{style}")
    
    if st.button("生成雷达图", key="radar_btn"):
        if not user_input:
            st.error("请输入有效的中文诗词文本。")
        else:
            # 清理文本
            cleaned_text = re.sub(r'[^\u4e00-\u9fa5，。！？、]', '', user_input)
            
            with st.spinner("正在分析..."):
                style_result, genre_probs, style_probs, imagery_contributions, purity_scores = predict_style(cleaned_text, thresholds3)
                
                # 显示结果
                st.success(f"诊断结果: {style_result}")
                
                # 显示风格纯净度雷达图
                st.subheader("风格纯净度分析")
                if purity_scores:
                    fig = plot_radar_chart(purity_scores)
                    st.plotly_chart(fig)
                    
                    # 显示数值
                    st.subheader("纯净度得分:")
                    for style, score in sorted(purity_scores.items(), key=lambda x: x[1], reverse=True):
                        st.text(f"{style}: {score:.4f}")
                else:
                    st.info("无法计算风格纯净度。")

with tab4:
    # 第四个标签页 - 诗词对比分析
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("诗词一")
        poem1 = st.text_area("内容:", height=150, key="poem1", value=example_poems[selected_example])
    
    with col2:
        st.subheader("诗词二")
        # 提供不同的示例
        second_example = "选择示例诗词..."
        if selected_example != "选择示例诗词..." and selected_example != list(example_poems.keys())[1]:
            second_example = list(example_poems.keys())[1]
        poem2 = st.text_area("内容:", height=150, key="poem2", value=example_poems[second_example])
    
    # 解释说明
    st.info("对比分析可以帮助您比较两首诗词在风格、意象等方面的差异。")
    
    # 交互式阈值调整
    with st.expander("高级选项: 阈值调整"):
        st.write("调整不同风格的识别阈值（值越低，越容易被识别为该风格）")
        col1, col2 = st.columns(2)
        
        thresholds4 = {}
        with col1:
            st.subheader("诗的风格阈值")
            for style in STYLE_LABELS['诗']:
                thresholds4.setdefault('诗', {})
                thresholds4['诗'][style] = st.slider(f"{style} 阈值", 0.1, 0.9, 0.4, 0.05, key=f"tab4_{style}")
        
        with col2:
            st.subheader("词的风格阈值")
            for style in STYLE_LABELS['词']:
                thresholds4.setdefault('词', {})
                thresholds4['词'][style] = st.slider(f"{style} 阈值", 0.1, 0.9, 0.4, 0.05, key=f"tab4_{style}")
    
    if st.button("对比分析", key="compare_btn"):
        if not poem1 or not poem2:
            st.error("请输入两首需要对比的诗词。")
        else:
            # 清理文本
            cleaned_poem1 = re.sub(r'[^\u4e00-\u9fa5，。！？、]', '', poem1)
            cleaned_poem2 = re.sub(r'[^\u4e00-\u9fa5，。！？、]', '', poem2)
            
            with st.spinner("正在对比分析..."):
                result1, genre_probs1, style_probs1, imagery1, purity1 = predict_style(cleaned_poem1, thresholds4)
                result2, genre_probs2, style_probs2, imagery2, purity2 = predict_style(cleaned_poem2, thresholds4)
                
                # 创建对比结果
                st.subheader("诊断结果对比")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**诗词一:**")
                    st.success(result1)

                with col2:
                    st.write("**诗词二:**")
                    st.success(result2)
                
                # 风格概率对比
                st.subheader("风格概率对比")
                
                # 确定两首诗词的体裁
                genre1 = result1.split(' - ')[0]
                genre2 = result2.split(' - ')[0]
                
                # 创建对比数据
                if genre1 == genre2:
                    # 同一体裁，直接对比
                    comp_data = pd.DataFrame({
                        '诗词一': list(style_probs1.values()),
                        '诗词二': list(style_probs2.values())
                    }, index=list(style_probs1.keys()))
                    
                    st.table(comp_data)
                    
                    # 雷达图对比
                    st.subheader("风格纯净度对比")
                    
                    categories = list(purity1.keys())
                    # 闭合雷达图
                    categories.append(categories[0])
                    
                    values1 = list(purity1.values())
                    values1.append(values1[0])
                    
                    values2 = list(purity2.values())
                    values2.append(values2[0])
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values1,
                        theta=categories,
                        fill='toself',
                        name='诗词一',
                        line_color='rgb(0, 102, 204)',
                        fillcolor='rgba(0, 102, 204, 0.3)'
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values2,
                        theta=categories,
                        fill='toself',
                        name='诗词二',
                        line_color='rgb(204, 0, 0)',
                        fillcolor='rgba(204, 0, 0, 0.3)'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0.9, 1]  # 将范围从0-1调整为0.9-1，以便更好地显示细微差异
                            )
                        ),
                        showlegend=True,
                        title="风格纯净度对比"
                    )
                    
                    st.plotly_chart(fig)
                else:
                    # 不同体裁，分开显示
                    st.write(f"**诗词一 ({genre1}):**")
                    st.bar_chart(style_probs1)
                    
                    st.write(f"**诗词二 ({genre2}):**")
                    st.bar_chart(style_probs2)
                
                # 意象对比分析
                st.subheader("关键意象对比")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**诗词一主要意象:**")
                    if imagery1:
                        words1 = [item["word"] for item in imagery1[:5]]
                        st.write(", ".join(words1))
                    else:
                        st.info("未找到显著意象")
                
                with col2:
                    st.write("**诗词二主要意象:**")
                    if imagery2:
                        words2 = [item["word"] for item in imagery2[:5]]
                        st.write(", ".join(words2))
                    else:
                        st.info("未找到显著意象")

with tab5:
    # 第五个标签页 - 使用指南
    st.header("诗词风格诊断系统使用指南")
    
    st.subheader("系统概述")
    st.write("""
    本系统基于深度学习技术，对中国古典诗词进行风格分类和诊断。系统采用层次化分类方法，先判断体裁（诗/词），再判断具体风格。
    同时提供意象分析、风格纯净度等深入分析功能。最新版本特别增强了对田园诗的识别能力和句式结构分析功能。
    """)
    
    st.subheader("使用方法")
    st.markdown("""
    1. **基本诊断**: 在"风格诊断"标签页中输入诗词内容，点击"诊断风格"按钮，系统将给出体裁和风格判断。
    2. **意象分析**: 在"意象分析"标签页中输入诗词，可分析每个关键词对不同风格的贡献度。
    3. **纯净度分析**: 在"纯净度雷达图"标签页中可生成诗词的风格纯净度雷达图，直观展示风格特征。
    4. **对比分析**: 在"诗词对比"标签页中可对两首诗词进行对比，分析风格和意象差异。
    5. **示例诗词**: 使用下拉菜单选择示例诗词，快速体验系统功能。
    6. **阈值调整**: 在每个标签页的"高级选项"中可调整风格识别阈值，降低阈值使系统更容易识别该风格，提高阈值则更严格。
    """)
    
    st.subheader("阈值调整说明")
    st.markdown("""
    阈值调整功能可以帮助您微调诗词风格的识别灵敏度：
    - **降低阈值**（如从0.4降至0.3）：系统更容易将诗词识别为该风格
    - **提高阈值**（如从0.4升至0.5）：系统对该风格的识别更加严格
    - **默认阈值**：所有风格的默认阈值为0.4
    
    当诗词同时具有多种风格特征时，调整阈值特别有用。例如，对于兼具田园诗和山水诗特征的作品，可以适当调低您认为更符合的风格阈值。
    
    当最高概率的风格低于其阈值时，系统会尝试识别为混合风格。
    """)
    
    st.subheader("支持的风格类型")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**诗的风格:**")
        st.write(", ".join(STYLE_LABELS['诗']))
        
    with col2:
        st.write("**词的风格:**")
        st.write(", ".join(STYLE_LABELS['词']))
    
    st.subheader("诊断准确性说明")
    st.info("""
    系统的诊断结果受多种因素影响，包括但不限于:
    - 输入文本的完整性和规范性
    - 诗词风格特征的明显程度
    - 深度学习模型的训练数据范围
    
    最新版本特别增强了对田园诗的识别能力，并改进了句式结构分析功能。纯净度雷达图的显示范围已优化为0.9-1，以便更好地展示细微差异。
    
    如对诊断结果有疑问，可尝试调整"高级选项"中的阈值，或参考意象分析和纯净度分析结果。
    """)
# 添加页脚
st.markdown("---")
st.markdown("© 诗词风格诊断系统 | 基于深度学习的中国古典诗词风格分层分类") 