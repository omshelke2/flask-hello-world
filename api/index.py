from flask import Flask
from transformers import pipeline

# Load the classification pipeline with the specified model
pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

# Classify a new sentence
sentence = "bura product hai"
result = pipe(sentence)

# Print the result
# print(result)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/new/<string:name>")
def myName(name):
    if(name=="omkar"):
        return "True"
    else:
        return "False"
    
# for returning the reult of the sentiment of the statement at the top
@app.route("/ai")
def ai():
    return result


@app.route("/resolveAI/<string:query>")
def resolveAI(query):
    answer = pipe(query)
    return answer

     

if __name__=="__main__":
    app.run(debug=True)
