from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sents = ['люблю собак',
   'люди нелюди так сказать',
   'самое высокое здание в мире было спроектировано с моего скипетра']


cv = CountVectorizer()
cv2_2 = CountVectorizer(ngram_range=(2,2))

X = cv.fit_transform(sents)
X = X.toarray()

print(X)
print(sorted(cv.vocabulary_.keys()))

tfidf = TfidfVectorizer()
transformed = tfidf.fit_transform(sents)