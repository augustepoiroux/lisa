import json
from functools import cmp_to_key
from pathlib import Path
from typing import Union

import numpy as np
import torch
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from transformers import AutoTokenizer, T5EncoderModel

ROOT_DIR = Path(__file__).parent.parent
DATA_EXTRACT_DIR = ROOT_DIR / "data_extract"
SCALA_DIR = ROOT_DIR / "src" / "main" / "scala"
MATHS_DIR = SCALA_DIR / "lisa" / "mathematics"

console = Console()


# Load a corpus of premises to retrieve from
# walk in the "data_extract" directory and import all the json data
premises = {}
for path in DATA_EXTRACT_DIR.rglob("*.json"):
    with open(path) as f:
        console.log(path)
        query_data = json.load(f)
        for name, query_data in query_data.items():
            if name in premises:
                console.log(f"\tDuplicate name detected: {name}", style="red")
                continue
            premises[name] = query_data
console.log(f"Loaded {len(premises)} premises")

# Walk in the "data_extract" directory and recover dependencies between file
file_dependencies = {}
for path in DATA_EXTRACT_DIR.rglob("*.json"):
    if path.name != "Axioms.json":
        code_path = SCALA_DIR / path.relative_to(DATA_EXTRACT_DIR).with_suffix(".scala")
        if not code_path.exists():
            console.log(f"Missing code file for {path}!")
            continue
        with open(code_path) as f:
            # take only the lines starting with "import lisa.mathematics"
            imports = list(
                {
                    line.split("import lisa.mathematics.")[1].split(".*")[0].strip()
                    for line in f.readlines()
                    if line.startswith("import lisa.mathematics") and line.endswith(".*\n")
                }
            )
            # extract the path from the import
            imports = [MATHS_DIR / Path(import_path.replace(".", "/") + ".scala") for import_path in imports]
            # check that all paths exist
            if not all(import_path.exists() for import_path in imports):
                console.log(f"Missing import path: {imports}!")
                continue
            # get relative paths
            imports = [str(Path(import_path).relative_to(ROOT_DIR.parent)) for import_path in imports]
            rel_code_path = str(Path(code_path).relative_to(ROOT_DIR.parent))
            file_dependencies[rel_code_path] = set(imports)

# flatten dependencies
already_flattened = set()


def flatten_dependencies(file: str) -> list[str]:
    if file in already_flattened:
        return file_dependencies[file]
    already_flattened.add(file)
    file_dependencies[file] |= {
        dependency for dep in file_dependencies[file] for dependency in flatten_dependencies(dep)
    }
    return file_dependencies[file]


for file in file_dependencies:
    flatten_dependencies(file)


def prepare_input(name: str) -> str:
    """Prepare the input for the model."""
    data = premises[name]
    return name + ": " + data["statement"]
    return data["statement"]
    if "declaration" in data:
        return data["declaration"]
    else:
        return name + ": " + data["statement"]


def cmp_str(s1: str, s2: str) -> int:
    return 1 if s1 > s2 else (-1 if s1 < s2 else 0)


def order_files(file_1: str, file_2: str) -> int:
    """Order files by their dependencies."""
    assert file_1 in file_dependencies and file_2 in file_dependencies
    if file_1 == file_2:
        return 0
    elif file_1 in file_dependencies[file_2]:
        return -1
    elif file_2 in file_dependencies[file_1]:
        return 1
    else:
        return cmp_str(file_1, file_2)


def order_premises(premise_1: str, premise_2: str) -> int:
    """Order premises by their file and line number."""
    premise_data_1 = premises[premise_1]
    premise_data_2 = premises[premise_2]
    if "file" in premise_data_1 and "file" in premise_data_2:
        if premise_data_1["file"] == premise_data_2["file"]:
            return premise_data_1["line"] - premise_data_2["line"]
        else:
            return order_files(premise_data_1["file"], premise_data_2["file"])
    elif "file" in premise_data_1:
        return 1
    elif "file" in premise_data_2:
        return -1
    else:
        return cmp_str(premise_1, premise_2)


# sort premises
premise_names_dep_sorted = sorted(premises.keys(), key=cmp_to_key(order_premises))
premise_name_to_idx = {name: i for i, name in enumerate(premise_names_dep_sorted)}
premise_inputs = [prepare_input(name) for name in premise_names_dep_sorted]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.log(f"Using device: {device}")

with console.status("Loading model..."):
    # put model on GPU
    model_name = "kaiyuy/leandojo-lean3-retriever-byt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device)


@torch.no_grad()
def encode(s: Union[str, list[str]]) -> torch.Tensor:
    """Encode texts into feature vectors."""
    if isinstance(s, str):
        s = [s]
        should_squeeze = True
    else:
        should_squeeze = False
    tokenized_s = tokenizer(s, return_tensors="pt", padding=True).to(device)
    hidden_state = model(tokenized_s.input_ids).last_hidden_state
    lens = tokenized_s.attention_mask.sum(dim=1)
    features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
    if should_squeeze:
        features = features.squeeze()
    return features


# Encoding premises
batch_size = 16
emb_dim = encode(prepare_input(premise_names_dep_sorted[0])).shape[0]
premise_embs = torch.zeros(len(premise_inputs), emb_dim).to(device)
for i in track(range(0, len(premise_inputs), batch_size), description="Encoding premises..."):
    premise_embs[i : i + batch_size] = encode(premise_inputs[i : i + batch_size])


@torch.no_grad()
def retrieve(state: str):
    """Rank the premises given a state."""
    state_emb = encode(state)
    return state_emb @ premise_embs.T


def pretty_print_retrieval(thm_name: str, scores: torch.Tensor, ground_truth: set = set(), top_k: int = 20):
    # sort the scores and the premises accordingly
    scores, sorted_indices = scores.sort(descending=True)

    # Table with top_k premises
    table = Table(title=f"Retrieved premises ({top_k=})", show_lines=True)
    table.add_column("Premise")
    table.add_column("Score", justify="center", vertical="middle")
    table.add_column("Rank", justify="center", vertical="middle")
    table.add_column("Ground truth", justify="center", vertical="middle")

    # Table with ground truth premises
    table_gt = Table(title="Ground truth", show_lines=True)
    table_gt.add_column("Premise")
    table_gt.add_column("Score", justify="center", vertical="middle")
    table_gt.add_column("Rank", justify="center", vertical="middle")

    rank = 0
    for i, s in zip(sorted_indices, scores.tolist()):
        ## skip the theorem itself and justifications that are defined after the theorem
        if i >= premise_name_to_idx[thm_name]:
            continue

        premise_name = premise_names_dep_sorted[i]
        rank += 1

        # adapt the color of the score to the value
        # 1 is green, 0 is red, in between is a gradient
        s_color = np.clip(s, 0, 1)
        color = f"rgb({int(255 * (1 - s_color))},{int(255 * s_color)},0)"
        if rank <= top_k:
            table.add_row(
                Group(Text(premise_name), Panel.fit(premises[premise_name]["statement"], style="rgb(100,100,100)")),
                f"{s:.3f}",
                str(rank),
                "✓" if i.item() in ground_truth else "",
                style=color + (" bold" if i.item() in ground_truth else ""),
            )
        if i.item() in ground_truth:
            table_gt.add_row(
                Group(Text(premise_name), Panel.fit(premises[premise_name]["statement"], style="rgb(100,100,100)")),
                f"{s:.3f}",
                str(rank),
                style=color + " bold",
            )
    console.print(table)
    console.print(table_gt)


while True:
    console.print("\nEnter a justification name to retrieve premises for:")
    thm_name = Prompt.ask("Justification name: ", default="existentialConjunctionWithClosedFormula")
    # thm_name = Prompt.ask("Justification name: ", default="uniqueExistentialEquivalenceDistribution")
    if thm_name not in premises:
        console.log(f"Justification {thm_name} not found", style="red")
        continue

    state = prepare_input(thm_name)
    # get indices of actual premises
    ground_truth = set()
    for imp in premises[thm_name]["imports"]:
        if imp["name"] in premises:
            ground_truth.add(premise_name_to_idx[imp["name"]])

    console.log("Justification:", style="bold")
    console.log(state, style="bold green")
    console.log(f"Nb justifications retrievable: {premise_name_to_idx[thm_name]}")
    scores = retrieve(state)
    pretty_print_retrieval(thm_name, scores, ground_truth)
