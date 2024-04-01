from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
class Features:
    def encode_target(df, target_col, encoded_target):
        """Create a new column 'sentiment_id' with encoded categories"""
        df[encoded_target] = df[target_col].factorize()[0]
        return df

    def tf_vectorize(self, df, input_text, max_words):
        """train tf vectorization to create features"""
        tfidf = TfidfVectorizer(max_features=max_words, sublinear_tf=True, min_df=5,
                                ngram_range=(1, 2),
                                stop_words='english')

        tfidf.fit_transform(df[input_text]).toarray()
        return tfidf

    def count_vector(self, df, input_text, max_words):
        """Train Count vectorization on train data"""
        countvect = CountVectorizer(max_features=max_words, min_df=5,
                                    ngram_range=(1, 2),
                                    stop_words='english')

        countvect.fit_transform(df[input_text]).toarray()
        return countvect
