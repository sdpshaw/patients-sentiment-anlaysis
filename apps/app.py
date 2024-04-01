import pandas as pd
from flask import request, Flask, render_template
from preprocessing.preprocess import Preprocessing
app = Flask(__name__)

app.route('/')


def home():
    return render_template('index.html')


app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    text = [x for x in request.form.values()]
    df = pd.DataFrame({"input_text":text[0]})
    obj = Preprocessing(df, 'input_text')
    df = obj.df


if __name__ == '__main__':
    app.run(debug=True)
