import json
from pathlib import Path

from rich.console import Console, Group
from rich.json import JSON
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

console = Console()

# Global prompt
lisa_intro_prompt = r"""LISA is a Scala-based formal system to prove theorems.
LISA’s foundations are based on very traditional (in the mathematical
community) foundational theory of all mathematics: First Order Logic,
expressed using Sequent Calculus (augmented with schematic symbols),
with axioms of Set Theory.

Here is a list of the main constructs of the language:

Common tactics:
- `Restate` is a tactic that reasons modulo ortholattices, a subtheory of boolean
algebra. Formally, it is very eﬀicient and can
prove a lot of simple propositional transformations, but not everything that
is true in classical logic. In particular, it can’t prove that (a∧b)∨(a∧c) ⇐⇒
a ∧ (b ∨ c) is true. It can however prove very limited facts involving equality
and quantifiers. Usage:
`have(statement) by Restate`
tries to justify statement by showing it is equivalent to True.
`have(statement) by Restate(premise)`
tries to justify statement by showing it is equivalent to the previously proven
premise.
- `Tautology` is a propositional solver based upon restate, but complete. It is
able to prove every formula inference that holds in classical propositional
logic. However, in the worst case its complexity can be exponential in the
size of the formula. Usage:
`have(statement) by Tautology`
Constructs a proof of statement, if the statement is true and a proof of it
using only classical propositional reasoning exists.
`have(statement) by Tautology.from(premise1, premise2,...)`
Construct a proof of statement from the previously proven premise1,
premise2,... using propositional reasoning.
- `RightForall` will generalize a statement by quantifying it over free variables.
For example,
```scala
have(P(x)) by ???
thenHave(∀(x, P(x))) by RightForall
```
Note that if the statement inside `have` has more than one formula, x cannot
appear (it cannot be free) in any formula other than P(x). It can also not
appear in any assumption.
- `InstantiateForall` does the opposite: given a universally quantified
statement, it will specialize it. For example:
```scala
have(∀(x, P(x))) by ???
thenHave(P(t)) by InstantiateForall
```
for any arbitrary term t.
- `Substitution` allows reasoning by substituting equal terms and equivalent
formulas. Usage:
`have(statement) by Substitution.ApplyRules(subst*)(premise)`
`subst*` is an arbitrary number of substitution. Each of those can be a
previously proven fact (or theorem or axiom), or a formula. They must all be
of the form s === t or A <=> B, otherwise the tactic will fail. The premise
is a previously proven fact. The tactic will try to show that statement can
be obtained from premise by applying the substitutions from subst. In its
simplest form,
```scala
val subst = have(s === t) by ???
have(P(s)) by ???
thenHave(P(t)) by Substitution.ApplyRules(subst)
```
Moreover, `Substitution` is also able to automatically unify and instan-
tiate subst rules. For example
```scala
val subst = have(g(x, y) === g(y, x)) by ???
have(P(g(3, 8))) by ???
thenHave(P(g(8, 3))) by Substitution.ApplyRules(subst)
```
If a subst is a formula rather than a proven fact, then it should be an
assumption in the resulting statement. Similarly, if one of the substitution
has an assumption, it should be in the resulting statement. For example,
```scala
val subst = have(A |- Q(s) <=> P(s)) by ???
have(Q(s) /\ s===f(t)) by ???
thenHave(A, f(t) === t |- P(s) /\ s===t)
.by Substitution.ApplyRules(subst, f(t) === t)
```

`assume` construct: this tells to LISA that the
assumed formula is understood as being implicitly on the left hand side of
every statement in the rest of the proof.
```scala
val fixedPointDoubleApplication = Theorem(
    ∀(x, P(x) ==> P(f(x))) |- P(x) ==> P(f(f(x)))
) {
    assume(∀(x, P(x) ==> P(f(x))))
    ...
}
```

Let's see two examples of proofs in LISA to get a better idea of how it works.
- Example 1:
```scala
val fixedPointDoubleApplication = Theorem(
    ∀(x, P(x) ==> P(f(x))) |- P(x) ==> P(f(f(x)))
) {
    assume(∀(x, P(x) ==> P(f(x))))
    val step1 = have(P(x) ==> P(f(x))) by InstantiateForall
    val step2 = have(P(f(x)) ==> P(f(f(x)))) by InstantiateForall
    have(thesis) by Tautology.from(step1, step2)
}
```
After the `assume` construct, we instantiate the quantified formula twice using a spe-
cialized tactic. We use `have` to state that a formula or
sequent is true (given the assumption inside assume), and that the proof of
this is produced by the tactic `InstantiateForall`. To be able to reuse intermediate steps at any
point later, we also assign the intermediates step to a variable.
Finally, the last line says that the conclusion of the theorem itself,
`thesis`, can be proven using the tactic `Tautology` and the two interme-
diate steps we reached. `Tautology` is a tactic that is able to do reasoning
with propositional connectives. It implements a complete decision procedure
for propositional logic.

- Example 2:
```scala
val emptySetIsASubset = Theorem(
    ∅ ⊆ x
) {
    have((y ∈ ∅) ==> (y ∈ x)) by Tautology.from(
    emptySetAxiom of (x := y))
    val rhs = thenHave (∀(y, (y ∈ ∅) ==> (y ∈ x))) by RightForall
    have(thesis) by Tautology.from(
    subsetAxiom of (x := ∅, y := x), rhs)
}
```
`thenHave` is similar to `have`, but it will automatically pass the previous statement to the tactic. Formally,
```scala
have(X) by Tactic1
thenHave (Y) by Tactic2
```
is equivalent to
```scala
val s1 = have(X) by Tactic1
have (Y) by Tactic2(s1)
```
`thenHave` allows us to not give a name to every step when we’re doing
linear reasoning.
The `of` keyword indicates the axiom (or step) is instantiated in a particular way. For example:
```emptySetAxiom // == !(x ∈ ∅)```
```emptySetAxiom of (x := y) // == !(y ∈ ∅)```"""


with console.status("Loading data..."):
    # walk in the "data_extract" directory and import all the json data
    # into a dictionary
    theorems, definitions, axioms = {}, {}, {}
    for path in Path("data_extract").rglob("*.json"):
        with open(path) as f:
            console.log(f"{path}...")
            query_data = json.load(f)
            for name, query_data in query_data.items():
                if name in theorems or name in definitions or name in axioms:
                    console.log(f"\tDuplicate name detected: {name}", style="red")
                    continue
                if query_data["kind"].lower() in ("theorem", "lemma", "corollary"):
                    theorems[name] = query_data
                elif "definition" in query_data["kind"].lower():
                    definitions[name] = query_data
                elif "axiom" in query_data["kind"].lower():
                    axioms[name] = query_data
            console.log(f"\tLoaded {len(theorems)} theorems, {len(definitions)} definitions, {len(axioms)} axioms")


class DependencyTree:
    def __init__(self, name: str, data: dict | None = None):
        self.name = name
        self.data = data
        self.children: list[DependencyTree] = []

    @classmethod
    def build_recursively(cls, name: str, data: dict | None = None):
        """Compute the definition dependencies tree for the query by recursively
        going through the dependencies."""

        def rec_dep(defs: dict, tree: DependencyTree):
            for dep in defs:
                if dep in definitions:
                    branch = tree.add(dep, definitions[dep])
                    rec_dep(definitions[dep]["definitions"], branch)
                else:
                    branch = tree.add(dep)

        tree = cls(name, data)
        if data is not None:
            rec_dep(data["definitions"], tree)
        return tree

    def add(self, name: str, data: dict | None = None):
        child = DependencyTree(name, data)
        self.children.append(child)
        return child

    def root_string(self) -> str:
        if self.data is None:
            return ""
        return self.data["docstring"] + "\n" + self.data["declaration"]

    def build_context(self, add_root: bool = True, is_root: bool = True) -> str:
        if is_root or "_visited" not in self.__dict__:
            self._visited = set()
        self._visited.add(self.name)
        prompt = "\n\n".join(
            child.build_context(True, False)
            for child in self.children
            if child.data is not None and child.name not in self._visited
        )
        if self.data is not None and add_root:
            if prompt != "":
                prompt += "\n\n"
            prompt += self.root_string()
        return prompt

    def __rich__(self):
        if self.data is not None:
            tree = Tree(
                Group(
                    Text(self.name, style="green"),
                    Panel.fit(f"Statement: {self.data['statement']}"),
                )
            )
        else:
            tree = Tree(self.name, style="red")

        for child in self.children:
            tree.add(child.__rich__())
        return tree


console.log("\n")
while True:
    # query = Prompt.ask("Query: ", default="pairReconstruction")
    # query = Prompt.ask("Query: ", default="relationImpliesRelationBetweenDomainAndRange")
    # query = Prompt.ask("Query: ", default="functionalMembership")
    query = Prompt.ask("Query: ", default="functionFromImpliesFunctional")

    if query in theorems:
        query_data = theorems[query]
    elif query in definitions:
        query_data = definitions[query]
    elif query in axioms:
        query_data = axioms[query]
    else:
        console.log("Query not found")
        continue
    break

console.log(Syntax(query_data["declaration"], "scala", word_wrap=True))

console.log("\nDependencies tree for the query:")
tree = DependencyTree.build_recursively(query, query_data)
console.log(tree)

generated_local_context = tree.build_context()
console.log(f"\nLocal code context generated for the query ({len(generated_local_context)} chars):")
console.log(Syntax(generated_local_context, "scala", line_numbers=True, word_wrap=True))

full_prompt = (
    lisa_intro_prompt
    + "\n\nWe are interested in proving a new theorem. In the following lines, a list of related definitions, axioms, and theorems is provided.\n"
    + tree.build_context(add_root=False)
    + "\n\nThe proof for the following theorem is missing. Please provide a proof in Lisa for it.\n"
    + tree.root_string()
)
console.log("Full prompt (for copy-paste):")  # output the string (in particular with the newlines escaped)
print(full_prompt)
