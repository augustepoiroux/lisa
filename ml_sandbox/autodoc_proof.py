import json
from pathlib import Path

import numpy as np
from rich.console import Console, Group
from rich.json import JSON
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

ROOT_DIR = Path(__file__).parent.parent
DATA_EXTRACT_DIR = ROOT_DIR / "data_extract"
SCALA_DIR = ROOT_DIR / "src" / "main" / "scala"
MATHS_DIR = SCALA_DIR / "lisa" / "mathematics"

console = Console()

# Global prompt
lisa_intro_prompt = r"""LISA is a Scala-based formal system to prove theorems.
For all the following proofs, provide clear and precise comments to understand the main steps."""


documented_theorems = {}
undocumented_theorems = {}
for path in DATA_EXTRACT_DIR.rglob("*.json"):
    with open(path) as f:
        console.log(path)
        query_data = json.load(f)
        for name, query_data in query_data.items():
            if name in documented_theorems:
                console.log(f"\tDuplicate name detected: {name}", style="red")
                continue
            if query_data["kind"].lower() in ("theorem", "lemma", "corollary"):
                if (
                    "code" in query_data
                    and query_data["code"] is not None
                    and query_data["code"] != ""
                    and "//" in query_data["code"]
                ):
                    documented_theorems[name] = query_data
                else:
                    undocumented_theorems[name] = query_data
console.log(f"Loaded {len(documented_theorems)} commented theorems")
console.log(f"Loaded {len(undocumented_theorems)} theorems without comments:")
console.log(undocumented_theorems.keys())


# sample a random theorem
sample = np.random.choice(list(documented_theorems.keys()))
sample = documented_theorems[sample]
console.print(Syntax(sample["code"], "scala", line_numbers=True, word_wrap=True))
