
from flask import Flask

from model import BartForMaskedLM

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == "__main__":
    model = BartForMaskedLM.load_from_checkpoint('lightning_logs/version_3/checkpoints/epoch=3-step=290267.ckpt')
    print(model.learning_rate)
    # prints the learning_rate you used in this checkpoint
    model.eval()

    app.run()