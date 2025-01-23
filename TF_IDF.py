from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

sents = ['люблю собак',
   'люди нелюди так сказать',
   'самое высокое здание в мире было спроектировано с моего скипетра']

tfidf = TfidfVectorizer()

transformed = tfidf.fit_transform(sents)

df = pd.DataFrame(transformed[0].T.todense(),
  index=tfidf.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print(df)