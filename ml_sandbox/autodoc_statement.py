import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.json import JSON

ROOT_DIR = Path(__file__).parent.parent
DATA_EXTRACT_DIR = ROOT_DIR / "data_extract"
SCALA_DIR = ROOT_DIR / "src" / "main" / "scala"
MATHS_DIR = SCALA_DIR / "lisa" / "mathematics"

console = Console()

# Global prompt
lisa_intro_prompt = r"""LISA is a Scala-based formal system to prove theorems.
For all the following justifications, provide a clear and precise documentation."""


justifications = {}
undocumented_justifications = {}
for path in DATA_EXTRACT_DIR.rglob("*.json"):
    with open(path) as f:
        console.log(path)
        query_data = json.load(f)
        for name, query_data in query_data.items():
            if name in justifications:
                console.log(f"\tDuplicate name detected: {name}", style="red")
                continue
            if "docstring" in query_data and query_data["docstring"] is not None and query_data["docstring"] != "":
                justifications[name] = query_data
            else:
                undocumented_justifications[name] = query_data
console.log(f"Loaded {len(justifications)} justifications with docstrings")
console.log(f"Loaded {len(undocumented_justifications)} justifications without docstrings:")
console.log(undocumented_justifications.keys())


def generate_random_prompt(k: int = 5):
    """Generate a random prompt for autodocuments"""
    # instantiate the chat prompt
    prompt = [{"role": "system", "content": lisa_intro_prompt}]
    # get k random justifications
    random_justifications = np.random.choice(list(justifications.keys()), size=k, replace=False)
    # add the justifications to the prompt
    for justification in random_justifications:
        prompt.append({"role": "user", "content": justifications[justification]["declaration"]})
        prompt.append({"role": "system", "content": justifications[justification]["docstring"]})
    return prompt


console.print(JSON.from_data(generate_random_prompt(5)))
