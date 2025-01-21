from sklearn.feature_extraction.text import CountVectorizer

sents = ['coronavirus is a highly infectious disease',
   'coronavirus affects older people the most',
   'older people are at high risk due to this disease']

cv = CountVectorizer()
cv2_2 = CountVectorizer(ngram_range=(2,2))


X = cv.fit_transform(sents)
X = X.toarray()

print(X)

print(sorted(cv.vocabulary_.keys()))