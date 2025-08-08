import os
from dotenv import load_dotenv
import dspy
from sklearn.metrics import classification_report
from src.dataset import get_liar
from src.dsp_modules import ModerationModule

# Load .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# Configure DSPy
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini", api_key=openai_key))

# Init moderation module
moderator = ModerationModule()

# Load dataset
dataset = get_liar(split="test")

true_labels = []
pred_labels = []

for example in dataset.select(range(20)):  # Limit for cost
    statement = example["statement"]
    gold_label = example["label"]
    pred_label, _ = moderator(statement)
    true_labels.append(gold_label)
    pred_labels.append(pred_label)

print(classification_report(true_labels, pred_labels))
