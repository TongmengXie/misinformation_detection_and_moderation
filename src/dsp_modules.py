import dspy
import json

class ModerationModule(dspy.Module):
    def __init__(self, examples_path="prompts/examples.json"):
        super().__init__()
        with open(examples_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)

        self.prompt_text = "\n".join(
            [
                f"Statement: {ex['statement']}\nLabel: {ex['label']}\nJustification: {ex['justification']}"
                for ex in self.examples
            ]
        )

        self.moderator = dspy.Predict(
            "statement -> label, justification"
        )

    def forward(self, statement):
        full_prompt = f"""
You are a content moderation system. 
Your task is to classify the following statement as 'true', 'false', or 'opinion'. 
Do not censor legitimate political debate.
Base your decision on factual accuracy and avoid over-blocking opinion-based content.

Here are some examples:
{self.prompt_text}

Now classify:
Statement: {statement}
Label:
Justification:
"""
        result = self.moderator(statement=full_prompt)
        return result.label, result.justification
