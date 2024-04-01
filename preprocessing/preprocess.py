import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Preprocessing:
    """
    It will contain all the preprocessing steps required for the text
    """
    def __init__(self, df, col):
        self.df = df
        self.col = col
        self.col_clean = col
        self.lower_case()
        self.tokenize_sentence()
        self.stemming()
        self.remove_non_alpha()
        self.remove_stopwords()
      def lower_case(self):
        # Lowercase the text
        self.df[self.col_clean] = self.df[self.col].str.lower()
      def tokenize_sentence(self):
        # Tokenize the text
        self.df[self.col_clean] = self.df[self.col].apply(word_tokenize)
      def remove_stopwords(self):
        # Remove stopwords
        self.df[self.col_clean] = self.df[self.col_clean].apply(lambda x: " ".join([char for char in x if char not in stopwords.words('english')]))
      def remove_non_alpha(self):
        # Remove punctuation and other no alphabetic characters from tokens
        self.df[self.col_clean] = self.df[self.col_clean].apply(lambda x: [char for char in x if char.isalpha()])
      def stemming(self):
        # Convert the tokenized words into its base form
        lm = WordNetLemmatizer()
        self.df[self.col_clean] = self.df[self.col_clean].apply(lambda x: [lm.lemmatize(word) for word in x])
