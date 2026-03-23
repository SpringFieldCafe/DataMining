import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1 读取数据
file_path = os.path.join(os.path.dirname(__file__), 'spam.csv')
data = pd.read_csv(file_path)
data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# 2 数据预处理
X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25)

# 3 模型训练，基于X_train和y_train进行训练，包括原始数据转化为词频向量，基于词频向量训练贝叶斯模型
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# 4 模型评估，基于X_test和y_test进行准确率评估
X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

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

# 预测测试邮件
test_mails_vectorized = vectorizer.transform(test_mails)
predictions = model.predict(test_mails_vectorized)

# 输出预测结果
print("\n测试邮件预测结果:")
for i, (mail, pred) in enumerate(zip(test_mails, predictions)):
    status = "垃圾邮件" if pred == 1 else "正常邮件"
    print(f"邮件 {i+1}: {status}")
    print(f"内容: {mail[:100]}..." if len(mail) > 100 else f"内容: {mail}")
    print()

