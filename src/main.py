import os
from dotenv import load_dotenv, find_dotenv
import dspy
from src.dataset import get_liar
from src.dsp_modules import ModerationModule
import dspy

# Load .env
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# Configure DSPy


lm = dspy.LM('openai/gpt-4o-mini', api_key=OPENAI_API_KEY)  # or rely on env var
dspy.configure(lm=lm)

# Init moderation module
moderator = ModerationModule()

# Load dataset
dataset = get_liar(split="train")

# Run on first 5 examples
for example in dataset.select(range(5)):
    statement = example["statement"]
    label, justification = moderator(statement)
    print(f"🗣️ {statement}\n📌 Label: {label}\n🧠 Reason: {justification}\n{'-'*50}")
