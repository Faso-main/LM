from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

"""Здесь вполне применима также концепция n-грамм. В частности, мы можем комбинировать слова в группы по 2, 3, 4 и более слов, чтобы сгенерировать окончательный набор признаков.
Вместе с n-граммами также существует ряд параметров, таких как min_df, max_df, max_features, sublinear_tf и т.д., с которыми можно поэкспериментировать. Если грамотно их настроить, можно значительно улучшить возможности модели."""

sents = ['люблю собак',
   'люди нелюди так сказать',
   'самое высокое здание в мире было спроектировано с моего скипетра']

tfidf = TfidfVectorizer()

transformed = tfidf.fit_transform(sents)

df = pd.DataFrame(transformed[0].T.todense(),
  index=tfidf.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print(df)