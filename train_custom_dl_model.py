import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np
from gensim.models import Word2Vec
import jieba
from torch.optim import AdamW
import os
import random
from tqdm import tqdm

# 配置
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 50  # 增加训练轮数
LEARNING_RATE = 1e-3  # 适当降低学习率以避免过拟合
DROPOUT_RATE = 0.4  # 增加dropout率以减少过拟合
WEIGHT_DECAY = 5e-5  # 增加L2正则化权重
PATIENCE = 12  # 增加早停耐心值
LR_SCHEDULER_FACTOR = 0.7  # 调整学习率衰减因子
LR_SCHEDULER_PATIENCE = 4  # 增加学习率调整耐心值
GENRE_LOSS_WEIGHT = 1.0  # 增加体裁分类损失权重，使模型更重视体裁分类
STYLE_LOSS_WEIGHT = 0.8  # 略微降低风格分类损失权重
EVAL_INTERVAL = 3  # 每3轮评估一次

# 设置随机种子以确保可重复性
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 创建models目录（如果不存在）
os.makedirs('models', exist_ok=True)

# 定义分层标签
# 第一层：体裁
GENRE_LABELS = ['诗', '词']

# 第二层：风格（按体裁分类）
STYLE_LABELS = {
    '诗': ['边塞诗', '山水诗', '田园诗', '咏物诗', '咏史诗', '送别诗'],
    '词': ['豪放词', '婉约词']
}

# 数据增强函数
def augment_text(text, num_aug=2):
    """增强的数据增强：随机删除、替换、交换字符，以及添加同义词替换"""
    augmented_texts = []
    words = list(text)
    
    for _ in range(num_aug):
        aug_words = words.copy()
        
        # 随机选择操作：删除、交换、同义词替换
        ops = ['delete', 'swap', 'synonym']
        op = random.choice(ops)
        
        if op == 'delete' and len(aug_words) > 10:
            # 随机删除1-3个字符
            for _ in range(random.randint(1, min(3, len(aug_words) // 10))):
                if len(aug_words) > 10:  # 确保文本不会太短
                    del_idx = random.randint(0, len(aug_words) - 1)
                    del aug_words[del_idx]
        
        elif op == 'swap' and len(aug_words) > 5:
            # 随机交换相邻字符
            for _ in range(random.randint(1, min(3, len(aug_words) // 10))):
                idx = random.randint(0, len(aug_words) - 2)
                aug_words[idx], aug_words[idx + 1] = aug_words[idx + 1], aug_words[idx]
        
        elif op == 'synonym' and len(aug_words) > 5:
            # 模拟同义词替换（对于汉字，我们简单地随机替换一些常见字）
            common_chars = '风云花月山水鸟雁天地日月星辰江河湖海林木竹石春夏秋冬雨雪霜露'
            for _ in range(random.randint(1, min(3, len(aug_words) // 10))):
                if len(aug_words) > 5:
                    idx = random.randint(0, len(aug_words) - 1)
                    if aug_words[idx] in common_chars:
                        # 替换为另一个常见字
                        replacement_idx = random.randint(0, len(common_chars) - 1)
                        aug_words[idx] = common_chars[replacement_idx]
        
        # 保证增强后的文本不会太短
        if len(aug_words) >= len(text) * 0.8:
            augmented_texts.append(''.join(aug_words))
    
    return augmented_texts

# 自定义Dataset
class PoetryDataset(Dataset):
    def __init__(self, texts, genre_labels, style_labels, word2vec_model, max_len):
        self.texts = texts
        self.genre_labels = genre_labels  # 体裁标签（诗/词）
        self.style_labels = style_labels  # 风格标签
        self.word2vec_model = word2vec_model
        self.max_len = max_len
        self.vector_size = word2vec_model.vector_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text_tokens = self.texts[item] # 此时texts已经是分词后的列表
        genre_label = self.genre_labels[item]
        style_label = self.style_labels[item]

        # 将分词后的文本转换为Word2Vec向量序列
        # 限制最大长度
        if len(text_tokens) > self.max_len:
            text_tokens = text_tokens[:self.max_len]
        
        # 创建序列向量
        sequence = []
        for token in text_tokens:
            if token in self.word2vec_model.wv:
                sequence.append(self.word2vec_model.wv[token])
            else:
                # 对于未知词，使用零向量
                sequence.append(np.zeros(self.vector_size))
        
        # 填充序列到最大长度
        padding_length = self.max_len - len(sequence)
        if padding_length > 0:
            sequence.extend([np.zeros(self.vector_size)] * padding_length)
        
        # 转换为张量
        sequence_tensor = torch.tensor(np.array(sequence), dtype=torch.float)
        
        return {
            'text': " ".join(text_tokens), # 返回原始文本，用于后续可能的调试或显示
            'features': sequence_tensor,
            'genre_label': torch.tensor(genre_label, dtype=torch.long),
            'style_label': torch.tensor(style_label, dtype=torch.float)
        }

# 数据加载函数
def load_data(filepath, genre_mlb=None, style_mlbs=None, is_train=True, augment=False):
    texts = []
    genre_labels = []
    style_labels = []
    tokenized_texts = [] # 存储分词后的文本
    print(f"加载数据: {filepath}...")
    
    # 如果文件不存在，创建示例数据
    if not os.path.exists(filepath):
        print(f"文件 {filepath} 不存在，创建示例数据...")
        create_sample_data(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
        for data in all_data:
            original_text = data['poem']
            original_labels = data['labels']
            
            # 确定体裁标签（诗/词）
            genre = '词' if any('词' in label for label in original_labels) else '诗'

            # 分词
            tokens = list(jieba.cut(original_text))

            texts.append(original_text)
            genre_labels.append(genre)
            style_labels.append(original_labels)
            tokenized_texts.append(tokens)
            
            if is_train and augment:
                # 应用数据增强
                augmented_texts = augment_text(original_text, num_aug=2)
                for aug_text in augmented_texts:
                    aug_tokens = list(jieba.cut(aug_text))
                    texts.append(aug_text)
                    genre_labels.append(genre)
                    style_labels.append(original_labels)
                    tokenized_texts.append(aug_tokens)

    # 处理体裁标签
    if genre_mlb is None:
        genre_mlb = MultiLabelBinarizer(classes=GENRE_LABELS)
        genre_mlb.fit([GENRE_LABELS])
    
    # 将体裁标签转换为索引（用于交叉熵损失）
    genre_indices = [GENRE_LABELS.index(label) for label in genre_labels]
    
    # 处理风格标签
    if style_mlbs is None:
        style_mlbs = {}
        for genre in GENRE_LABELS:
            style_mlbs[genre] = MultiLabelBinarizer(classes=STYLE_LABELS[genre])
            style_mlbs[genre].fit([STYLE_LABELS[genre]])
    
    # 将风格标签二值化，每种体裁都有自己的二值化器
    binarized_style_labels = []
    for i, labels in enumerate(style_labels):
        genre = genre_labels[i]
        # 过滤出属于当前体裁的风格标签
        filtered_labels = [label for label in labels if (genre == '诗' and label in STYLE_LABELS['诗']) or 
                                                      (genre == '词' and label in STYLE_LABELS['词'])]
        # 如果没有匹配的标签，使用空列表
        if not filtered_labels:
            filtered_labels = []
        
        # 二值化风格标签
        bin_labels = style_mlbs[genre].transform([filtered_labels])[0]
        binarized_style_labels.append(bin_labels)

    return tokenized_texts, genre_indices, binarized_style_labels, genre_mlb, style_mlbs

# 创建示例数据
def create_sample_data(filepath):
    sample_data = [
        # 诗 - 边塞诗
        {"poem": "烽火城西百尺楼，黄昏独坐海风秋。更吹羌笛关山月，无那金闺万里愁。", "labels": ["边塞诗"]},
        {"poem": "秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。", "labels": ["边塞诗"]},
        {"poem": "大漠孤烟直，长河落日圆。萧关逢候骑，都护在燕然。", "labels": ["边塞诗"]},
        {"poem": "胡马依北风，越鸟巢南枝。", "labels": ["边塞诗"]},
        {"poem": "戍鼓断人行，边秋一雁声。露从今夜白，月是故乡明。", "labels": ["边塞诗"]},
        
        # 诗 - 山水诗
        {"poem": "江流天地外，山色有无中。郡邑浮前浦，波澜动远空。", "labels": ["山水诗"]},
        {"poem": "空山新雨后，天气晚来秋。明月松间照，清泉石上流。", "labels": ["山水诗"]},
        {"poem": "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。", "labels": ["山水诗"]},
        {"poem": "山中相送罢，日暮掩柴扉。春草明年绿，王孙归不归。", "labels": ["山水诗"]},
        {"poem": "泉眼无声惜细流，树阴照水爱晴柔。小荷才露尖尖角，早有蜻蜓立上头。", "labels": ["山水诗"]},
        
        # 诗 - 田园诗
        {"poem": "种豆南山下，草盛豆苗稀。晨兴理荒秽，带月荷锄归。道狭草木长，夕露沾我衣。衣沾不足惜，但使愿无违。", "labels": ["田园诗"]},
        {"poem": "莫笑农家腊酒浑，丰年留客足鸡豚。山重水复疑无路，柳暗花明又一村。", "labels": ["田园诗"]},
        {"poem": "绿遍山原白满川，子规声里雨如烟。乡村四月闲人少，才了蚕桑又插田。", "labels": ["田园诗"]},
        {"poem": "草长莺飞二月天，拂堤杨柳醉春烟。儿童散学归来早，忙趁东风放纸鸢。", "labels": ["田园诗"]},
        {"poem": "田家少闲月，五月人倍忙。夜来南风起，小麦覆陇黄。", "labels": ["田园诗"]},
        
        # 诗 - 咏物诗
        {"poem": "一叶渔船两小童，收篙停棹坐船中。怪生无雨都张伞，不是遮头是使风。", "labels": ["咏物诗"]},
        {"poem": "墙角数枝梅，凌寒独自开。遥知不是雪，为有暗香来。", "labels": ["咏物诗"]},
        {"poem": "竹外桃花三两枝，春江水暖鸭先知。蒌蒿满地芦芽短，正是河豚欲上时。", "labels": ["咏物诗"]},
        {"poem": "稻花香里说丰年，听取蛙声一片。七八个星天外，两三点雨山前。", "labels": ["咏物诗"]},
        {"poem": "远看山有色，近听水无声。春去花还在，人来鸟不惊。", "labels": ["咏物诗"]},
        
        # 诗 - 咏史诗
        {"poem": "周公恐惧流言日，王莽谦恭未篡时。向使当初身便死，一生真伪复谁知。", "labels": ["咏史诗"]},
        {"poem": "商女不知亡国恨，隔江犹唱后庭花。", "labels": ["咏史诗"]},
        {"poem": "三顾频烦天下计，两朝开济老臣心。出师未捷身先死，长使英雄泪满襟。", "labels": ["咏史诗"]},
        {"poem": "杜陵有布衣，老大意转拙。许身一何愚，窃比稷与契。", "labels": ["咏史诗"]},
        {"poem": "前不见古人，后不见来者。念天地之悠悠，独怆然而涕下。", "labels": ["咏史诗"]},
        
        # 诗 - 送别诗
        {"poem": "孤帆远影碧空尽，唯见长江天际流。", "labels": ["送别诗"]},
        {"poem": "海内存知己，天涯若比邻。无为在歧路，儿女共沾巾。", "labels": ["送别诗"]},
        {"poem": "莫愁前路无知己，天下谁人不识君。", "labels": ["送别诗"]},
        {"poem": "桃花潭水深千尺，不及汪伦送我情。", "labels": ["送别诗"]},
        {"poem": "劝君更尽一杯酒，西出阳关无故人。", "labels": ["送别诗"]},
        
        # 词 - 豪放词
        {"poem": "大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。乱石穿空，惊涛拍岸，卷起千堆雪。江山如画，一时多少豪杰。", "labels": ["豪放词"]},
        {"poem": "乐游原上清秋节，咸阳古道音尘绝。音尘绝，西风残照，汉家陵阙。", "labels": ["豪放词"]},
        {"poem": "八百里分麾下炙，五十弦翻塞外声。沙场秋点兵，马作的卢飞快，弓如霹雳弦惊。了却君王天下事，嬴得生前身后名。可怜白发生。", "labels": ["豪放词"]},
        {"poem": "醉里挑灯看剑，梦回吹角连营。八百里分麾下炙，五十弦翻塞外声。沙场秋点兵，马作的卢飞快，弓如霹雳弦惊。", "labels": ["豪放词"]},
        {"poem": "怒发冲冠，凭栏处、潇潇雨歇。抬望眼，仰天长啸，壮怀激烈。三十功名尘与土，八千里路云和月。", "labels": ["豪放词"]},
        
        # 词 - 婉约词
        {"poem": "昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧。知否，知否？应是绿肥红瘦。", "labels": ["婉约词"]},
        {"poem": "寻寻觅觅，冷冷清清，凄凄惨惨戚戚。乍暖还寒时候，最难将息。", "labels": ["婉约词"]},
        {"poem": "一别都门三改火，天涯踏尽红尘。依然一笑作春温。无波真古井，有节是秋筠。", "labels": ["婉约词"]},
        {"poem": "红藕香残玉簟秋。轻解罗裳，独上兰舟。云中谁寄锦书来，雁字回时，月满西楼。", "labels": ["婉约词"]},
        {"poem": "多情只有春庭月，犹为离人照落花。", "labels": ["婉约词"]}
    ]
    
    # 保存到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # 创建测试数据 - 使用不同的样本
    test_filepath = filepath.replace('train', 'test')
    
    # 为测试集选择不同的样本
    test_data = []
    # 从每个类别中选择2个样本
    for i in range(0, len(sample_data), 5):
        test_data.extend(sample_data[i:i+2])
    
    with open(test_filepath, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"示例数据已创建: {filepath} 和 {test_filepath}")

# 定义分层分类模型
class HierarchicalClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, genre_output_size, style_output_sizes):
        super(HierarchicalClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.genre_output_size = genre_output_size
        self.style_output_sizes = style_output_sizes
        
        # Transformer编码器层 - 增加层数和头数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=8,  # 增加到8个注意力头
            dim_feedforward=hidden_size*4,  # 增加前馈网络大小
            dropout=DROPOUT_RATE,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)  # 增加到3层
        
        # 注意力层 - 多头自注意力
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征转换层 - 增加层数和复杂度
        self.feature_fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 添加层归一化
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 添加层归一化
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # 体裁分类层（诗/词）- 更复杂的分类头
        self.genre_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 添加层归一化
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 2, genre_output_size)
        )
        
        # 每种体裁的风格分类层 - 更复杂的分类头
        self.style_classifiers = nn.ModuleDict({
            genre: nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),  # 添加层归一化
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(hidden_size // 2, style_output_size)
            )
            for genre, style_output_size in style_output_sizes.items()
        })
    
    def forward(self, features):
        # 使用Transformer编码器处理序列
        transformer_output = self.transformer_encoder(features)
        
        # 应用注意力机制
        attention_weights = self.attention(transformer_output)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(attention_weights * transformer_output, dim=1)  # [batch_size, input_size]
        
        # 转换特征
        features = self.feature_fc(context_vector)  # [batch_size, hidden_size]
        
        # 体裁分类
        genre_logits = self.genre_classifier(features)  # [batch_size, genre_output_size]
        
        # 风格分类（对每种体裁）
        style_logits = {}
        for genre, classifier in self.style_classifiers.items():
            style_logits[genre] = classifier(features)  # [batch_size, style_output_size]
        
        return genre_logits, style_logits

# 自定义收集函数，处理不同长度的风格标签
def custom_collate_fn(batch):
    # 提取每个样本的各个字段
    texts = [item['text'] for item in batch]
    
    # 所有特征现在都是相同长度的序列，可以直接堆叠
    features = torch.stack([item['features'] for item in batch])
    
    genre_labels = torch.stack([item['genre_label'] for item in batch])
    
    # 对于风格标签，我们需要特殊处理
    # 由于每个样本的风格标签长度可能不同，我们不能直接使用stack
    # 相反，我们将它们保持为列表形式
    style_labels = [item['style_label'] for item in batch]
    
    return {
        'text': texts,
        'features': features,
        'genre_label': genre_labels,
        'style_label': style_labels
    }

# 训练函数
def train_model(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    # 设置进度条
    pbar = tqdm(data_loader, total=len(data_loader))
    
    for batch in pbar:
        # 获取数据
        features = batch['features'].to(device)
        genre_labels = batch['genre_label'].to(device)
        style_labels = [item.to(device) for item in batch['style_label']]
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        genre_logits, style_logits = model(features)
        
        # 计算体裁分类的损失（交叉熵损失）
        # 设置权重以平衡诗和词的样本数量差异
        genre_counts = torch.bincount(genre_labels)
        if len(genre_counts) > 1:  # 确保批次中有多个类别
            genre_weights = 1.0 / genre_counts.float()
            genre_weights = genre_weights / genre_weights.sum()  # 归一化
            genre_weights = genre_weights.to(device)
            genre_loss = nn.CrossEntropyLoss(weight=genre_weights)(genre_logits, genre_labels)
        else:
            genre_loss = nn.CrossEntropyLoss()(genre_logits, genre_labels)
        
        # 计算风格分类的损失（多标签二元交叉熵）
        # 对于'诗'和'词'类别，分别计算风格分类损失
        style_loss = 0
        for i, label in enumerate(genre_labels):
            genre = '诗' if label.item() == 0 else '词'
            style_logit = style_logits[genre][i].unsqueeze(0)  # [1, num_style_classes]
            style_target = style_labels[i].unsqueeze(0)  # [1, num_style_classes]
            
            # 计算正样本权重 - 为稀有类别提供更高的权重
            if style_target.sum() > 0:
                pos_weight = torch.ones([style_target.size(1)], device=device) * 3.0  # 默认权重
                # 为咏物诗、咏史诗和送别诗提供更高的权重
                if genre == '诗':
                    style_names = STYLE_LABELS['诗']
                    for idx, name in enumerate(style_names):
                        if name in ['咏物诗', '咏史诗', '送别诗']:
                            pos_weight[idx] = 5.0  # 增加这些类别的权重
            else:
                pos_weight = torch.ones([style_target.size(1)], device=device) * 3.0
            
            curr_style_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(style_logit, style_target)
            style_loss += curr_style_loss
        
        # 求平均以减轻批次大小的影响
        style_loss = style_loss / len(genre_labels)
        
        # 总损失 - 加权合并
        total_batch_loss = GENRE_LOSS_WEIGHT * genre_loss + STYLE_LOSS_WEIGHT * style_loss
        
        # 反向传播
        total_batch_loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        total_loss += total_batch_loss.item()
        
        # 更新进度条
        pbar.set_description(f"Loss: {total_batch_loss.item():.4f}")
    
    # 学习率调整
    scheduler.step(total_loss / len(data_loader))
    
    return total_loss / len(data_loader)

# 评估函数
def evaluate_model(model, data_loader, device, genre_mlb, style_mlbs):
    model.eval()
    
    # 存储预测结果和真实标签
    all_genre_preds = []
    all_genre_labels = []
    all_style_preds = {genre: [] for genre in GENRE_LABELS}
    all_style_labels = {genre: [] for genre in GENRE_LABELS}
    
    with torch.no_grad():
        for data in data_loader:
            features = data['features'].to(device)
            genre_labels = data['genre_label'].to(device)
            style_labels = data['style_label']
            
            # 前向传播
            genre_logits, style_logits = model(features)
            
            # 体裁预测
            genre_preds = torch.argmax(genre_logits, dim=1).cpu().numpy()
            
            # 记录体裁预测和真实标签
            all_genre_preds.extend(genre_preds)
            all_genre_labels.extend(genre_labels.cpu().numpy())
            
            # 风格预测 - 使用sigmoid激活
            for i, genre_idx in enumerate(genre_labels):
                genre = GENRE_LABELS[genre_idx.item()]
                style_logit = style_logits[genre][i]
                
                # 使用sigmoid获取概率分数
                style_prob = torch.sigmoid(style_logit).cpu().numpy()
                
                # 为不同风格设置不同的阈值
                if genre == '诗':
                    # 为不同的诗歌风格设置自定义阈值
                    thresholds = {
                        '边塞诗': 0.35,
                        '山水诗': 0.3,
                        '田园诗': 0.35,
                        '咏物诗': 0.4,  # 提高咏物诗的阈值以提高精确率
                        '咏史诗': 0.4,  # 提高咏史诗的阈值以提高精确率
                        '送别诗': 0.4   # 提高送别诗的阈值以提高精确率
                    }
                    # 应用不同的阈值
                    style_pred = np.zeros_like(style_prob, dtype=int)
                    for j, style_name in enumerate(STYLE_LABELS[genre]):
                        threshold = thresholds.get(style_name, 0.3)
                        style_pred[j] = int(style_prob[j] > threshold)
                else:
                    # 对词使用统一阈值
                    threshold = 0.35
                    style_pred = (style_prob > threshold).astype(int)
                
                # 记录风格预测和真实标签
                all_style_preds[genre].append(style_pred)
                all_style_labels[genre].append(style_labels[i].cpu().numpy())

    # 评估体裁分类性能
    genre_accuracy = accuracy_score(all_genre_labels, all_genre_preds)
    genre_report = classification_report(
        all_genre_labels, 
        all_genre_preds,
        target_names=GENRE_LABELS,
        digits=2
    )
    
    # 评估各体裁下的风格分类性能
    style_reports = {}
    for genre in GENRE_LABELS:
        if len(all_style_preds[genre]) == 0:
            style_reports[genre] = f"没有{genre}的样本进行评估"
            continue
            
        # 将列表转换为数组
        style_preds = np.array(all_style_preds[genre])
        style_labels = np.array(all_style_labels[genre])
        
        # 如果没有样本，跳过评估
        if style_preds.shape[0] == 0:
            style_reports[genre] = f"没有{genre}的样本进行评估"
            continue
            
        # 多标签分类报告
        try:
            style_reports[genre] = classification_report(
                style_labels, 
                style_preds,
                target_names=STYLE_LABELS[genre],
                digits=2,
                zero_division=0
            )
        except Exception as e:
            style_reports[genre] = f"评估{genre}风格分类时出错: {str(e)}"
    
    return genre_report, style_reports

def main():
    # 加载训练和测试数据
    X_train_raw, y_train_genre, y_train_style, genre_mlb, style_mlbs = load_data(
        'labeled_train.json', is_train=True, augment=True  # 启用数据增强
    )
    X_test_raw, y_test_genre, y_test_style, _, _ = load_data(
        'labeled_test.json', genre_mlb=genre_mlb, style_mlbs=style_mlbs, is_train=False, augment=False
    )
    
    # 保存标签二值化器
    joblib.dump(genre_mlb, 'genre_label_binarizer.pkl')
    for genre, mlb in style_mlbs.items():
        joblib.dump(mlb, f'style_label_binarizer_{genre}.pkl')
    
    print("标签二值化器已保存")

    # 训练 Word2Vec 模型
    print("训练 Word2Vec 模型...")
    # 增加vector_size和window大小以获取更丰富的词向量
    word2vec_model = Word2Vec(sentences=X_train_raw, vector_size=128, window=7, min_count=1, workers=4, sg=1)
    word2vec_model.save("word2vec_model.bin")
    print("Word2Vec 模型训练完成并保存为 word2vec_model.bin。")

    # 准备数据集
    train_dataset = PoetryDataset(
        X_train_raw, y_train_genre, y_train_style, word2vec_model, MAX_LEN
    )
    test_dataset = PoetryDataset(
        X_test_raw, y_test_genre, y_test_style, word2vec_model, MAX_LEN
    )

    # 数据加载器，使用自定义的收集函数
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # 创建模型
    input_size = word2vec_model.vector_size
    hidden_size = 256  # 增加隐藏层大小
    genre_output_size = len(GENRE_LABELS)
    style_output_sizes = {genre: len(styles) for genre, styles in STYLE_LABELS.items()}
    
    model = HierarchicalClassifier(
        input_size, hidden_size, genre_output_size, style_output_sizes
    )
    model = model.to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)

    # 训练循环
    print("开始训练...")
    best_loss = float('inf')
    patience = PATIENCE  # 早停的耐心值
    no_improve = 0  # 没有改善的轮数
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # 训练
        train_loss = train_model(model, train_data_loader, optimizer, scheduler, device)
        print(f"训练损失: {train_loss:.4f}")
        
        # 评估
        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == 0:  # 每3轮评估一次
            genre_report, style_reports = evaluate_model(
                model, test_data_loader, device, genre_mlb, style_mlbs
            )
            print("\n阶段性评估 - 体裁分类报告:")
            print(genre_report)
            
            for genre, report in style_reports.items():
                print(f"\n阶段性评估 - {genre}风格分类报告:")
                print(report)

        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'models/hierarchical_model.pth')
            print("保存最佳模型")
            no_improve = 0  # 重置没有改善的轮数
        else:
            no_improve += 1
            
        # 早停
        if no_improve >= patience:
            print(f"训练损失连续{patience}轮未改善，提前停止训练")
            break
    
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load('models/hierarchical_model.pth'))
    model.eval()
    
    # 评估
    genre_report, style_reports = evaluate_model(
        model, test_data_loader, device, genre_mlb, style_mlbs
    )
    
    print("\n体裁分类报告:")
    print(genre_report)
    
    for genre, report in style_reports.items():
        print(f"\n{genre}风格分类报告:")
        print(report)
    
    print("训练完成!")

if __name__ == "__main__":
    main() 