import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas.util.testing as tm
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import early_stopping
from lightgbm import log_evaluation
from sklearn.metrics import accuracy_score
accuracy_score(train['label'],np.argmax(lgb_oof,axis=1))

train = pd.read_csv('返乡发展人群预测/dataTrain.csv')
no_label = pd.read_csv('返乡发展人群预测/dataNoLabel.csv')
A = pd.read_csv('返乡发展人群预测/dataA.csv')

print(train.head())
print(no_label.head())
print(A.head())
print(train['label'].value_counts())
train['label'].value_counts().plot(kind='bar')
data=pd.concat([train, A],axis=0).reset_index(drop=True)
data=data.fillna('NAN')
print(data)

lbls={}
features=data.columns[1:-1]
print(len(features))
for col in tqdm(features):
    lbl=LabelEncoder()
    lbl.fit(data[col])
    data[col]=lbl.transform(data[col])
    
train, test = data[:len(train)], data[len(train):]
print(train)
print(test)

# 排除特征
# id等肯定是要排除的
feature_names = list(
    filter(
        lambda x: x not in ['id','label'],
        train.columns))
# label转为int类型
train['label']=train['label'].apply(lambda i:int(i))

def lgb_model(train, target, test, k):
    feats = [f for f in train.columns if f not in ['lable',  'url', 'url_count']]

    print('Current num of features:', len(feats))

    oof_probs = np.zeros((train.shape[0],2))
    output_preds = 0
    offline_score = []
    feature_importance_df = pd.DataFrame()
    parameters = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_error',
        'num_class': 2,
        'num_leaves': 31,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 15,
        'verbose': -1,
        'nthread': 4,
        'max_depth': 7
    }

    seeds = [2020]
    for seed in seeds:
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for i, (train_index, test_index) in enumerate(folds.split(train, target)):
            train_y, test_y = target.iloc[train_index], target.iloc[test_index]
            train_X, test_X = train[feats].iloc[train_index, :], train[feats].iloc[test_index, :]

            dtrain = lgb.Dataset(train_X,
                                 label=train_y)
            dval = lgb.Dataset(test_X,
                               label=test_y)
            lgb_model = lgb.train(
                parameters,
                dtrain,
                num_boost_round=20000,
                valid_sets=[dval],
                callbacks=[early_stopping(100), log_evaluation(100)],
            )
            oof_probs[test_index] = lgb_model.predict(test_X[feats], num_iteration=lgb_model.best_iteration) / len(
                seeds)
            offline_score.append(lgb_model.best_score['valid_0']['multi_error'])
            output_preds += lgb_model.predict(test[feats],
                                              num_iteration=lgb_model.best_iteration) / folds.n_splits / len(seeds)
            print(offline_score)
            # feature importance
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='gain')
            fold_importance_df["fold"] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('OOF-MEAN-AUC:%.6f, OOF-STD-AUC:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    print('feature importance:')
    print(feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False).head(50))

    return output_preds, oof_probs, np.mean(offline_score), feature_importance_df

print('开始模型训练train')
lgb_preds, lgb_oof, lgb_score, feature_importance_df = lgb_model(train=train[feature_names],
                                                                 target=train['label'],
                                                                 test=test[feature_names], k=10)

# 读取提交格式
example_A = pd.read_csv('返乡发展人群预测/submit_example_A.csv')
print(example_A.head())

# 获取最大概率标签
example_A['label']=np.argmax(lgb_preds,axis=1)
print(example_A['label'].value_counts())

# 保存
example_A.to_csv('sub.csv',index=None)
