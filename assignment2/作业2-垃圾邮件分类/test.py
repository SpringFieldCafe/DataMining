import pandas as pd
from sklearn.model_selection import train_test_split

# 1 读取数据
data = pd.read_csv('D:/spam.csv')
data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# 2 数据预处理
X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25)

# 3 模型训练，基于X_train和y_train进行训练，包括原始数据转化为词频向量，基于词频向量训练贝叶斯模型

# 4 模型评估，基于X_test和y_test进行准确率评估

# 5 结果验证，基于训练模型评估新测试集结果
test_mails = [
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flying start? 2 find out free, txt HORO followed by ur '
    'star sign, e. g. HORO ARIES',
    'Had your mobile 11 months or more? U R entitled to Update to the latest ',
    'WINNER!! As a valued network customer you have been selected to receivea'
    ' £900 prize reward! To claim call 09061701461. '
    'Claim code KL341. Valid 12 hours only.',
    'Had your mobile 11 months or more? U R entitled to Update to th'
    'e latest colour mobiles with camera for Free! Call The Mobile U'
    'pdate Co FREE on 08002986030',
    'I am gonna be home soon and i do not want to talk about this stuff anymore tonight, k? I have cried enough today.',
    'I have been searching for the right words to thank you for this breather. '
    'I promise i wont take your help for granted and will fulfil my promise.'
    ' You have been wonderful and a blessing at all times.',
    'I HAVE A DATE ON SUNDAY WITH WILL!!',
    'Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet.',
    'Thanks for your subscription to Ringtone UK '
    'your mobile will be charged £5/month Please confirm by replying YES or NO. '
    'If you reply NO you will not be charged'
]
