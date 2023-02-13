from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

@app.route("/prompt", methods=["GET"])
def prompt():
    initial = request.args.get("initial", 'a landscape')
    tokenizer = AutoTokenizer.from_pretrained("token")
    model = AutoModelForCausalLM.from_pretrained("model")
    init_tok = tokenizer.encode(initial.lower(), return_tensors='pt')
    prompt = model.generate(init_tok, do_sample=True, max_new_tokens=150)
    return tokenizer.decode(prompt[0], skip_special_tokens=True)

if __name__ == "__main__":
    app.run()