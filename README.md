# LLM-Powered Content Moderation (Minimal PoC)

This is a minimal proof-of-concept for a content moderation system using **DSPy** + **GPT-4o-mini** to detect misinformation while preserving legitimate political debate.

## Notes
Current exmaple data pieces: Treating "if specfic context and qunantifiable data is existent" as a signal for misformation/non-misinformation. "Misrepresents the context adn intetnt" seems to be another signal.

Experiments to be done: a. find datasets to control for each signal or composition of them: 1. lacking of specific context () 2. misinterpretation 3. sychphancy ()...

b. perform moderation

Current limitation: LLM-as-a-judge has 1. bias in political views and 2. hallucination (factuality)

## Getting Started

1. Clone repo and install requirements:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run:
```bash
python src/main.py
```

4. Evaluate:
```bash
python src/eval.py
```

## Structure
- `src/dataset.py`: Loads LIAR dataset.
- `src/dsp_modules.py`: DSPy few-shot moderation module.
- `src/main.py`: Entry point to run pipeline.
- `src/eval.py`: Evaluation script for accuracy/F1.
- `prompts/examples.json`: Few-shot examples.
- `.env`: Environment variables (not committed).

## Dataset
Uses the [LIAR dataset](https://huggingface.co/datasets/liar) for political truthfulness classification.

