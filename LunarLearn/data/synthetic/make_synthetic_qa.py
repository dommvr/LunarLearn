from LunarLearn.data.synthetic.utils import _get_rng


def make_synthetic_qa(
    n_samples=2000,
    n_people_range=(2, 4),
    max_count=9,
    entities=("apples", "bananas", "coins"),
    names=("Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi"),
    tasks=("who_more", "who_less", "difference", "total", "compare_yesno"),
    shuffle=True,
    random_state=None,
):
    """
    Template-based QA without huge datasets.
    Output: list of dicts: {"context", "question", "answer"} (strings)

    Tasks:
      - who_more: "Who has more apples?"
      - who_less: "Who has fewer bananas?"
      - difference: "How many more apples does Alice have than Bob?"
      - total: "How many apples do they have in total?"
      - compare_yesno: "Does Alice have more apples than Bob?"
    """
    rng = _get_rng(random_state)

    def pick_unique(k):
        idx = rng.choice(len(names), size=(k,), replace=False)
        return [names[int(i)] for i in idx.tolist()]

    samples = []

    for _ in range(n_samples):
        n_people = int(rng.randint(int(n_people_range[0]), int(n_people_range[1]) + 1))
        people = pick_unique(n_people)

        item = entities[int(rng.randint(0, len(entities)))]
        counts = {p: int(rng.randint(0, max_count + 1)) for p in people}

        # Build context
        # Example: "Alice has 3 apples. Bob has 2 apples."
        facts = [f"{p} has {counts[p]} {item}." for p in people]
        context = " ".join(facts)

        task = tasks[int(rng.randint(0, len(tasks)))]

        # Ensure we can answer unambiguously (ties are annoying for "who more/less")
        if task in ("who_more", "who_less"):
            # if tie, tweak one random person
            vals = [counts[p] for p in people]
            if len(set(vals)) == 1:
                # all equal; nudge one
                p0 = people[int(rng.randint(0, n_people))]
                counts[p0] = min(max_count, counts[p0] + 1)
                facts = [f"{p} has {counts[p]} {item}." for p in people]
                context = " ".join(facts)

        if task == "who_more":
            question = f"Who has more {item}?"
            # pick max; if tie, pick first by order in people (rare after tweak)
            winner = max(people, key=lambda p: counts[p])
            answer = winner

        elif task == "who_less":
            question = f"Who has fewer {item}?"
            loser = min(people, key=lambda p: counts[p])
            answer = loser

        elif task == "difference":
            # pick two distinct people
            a, b = people[0], people[1] if n_people > 1 else (people[0], people[0])
            # if somehow only 1 person, make it trivial
            question = f"How many more {item} does {a} have than {b}?"
            diff = counts[a] - counts[b]
            answer = str(diff)

        elif task == "total":
            question = f"How many {item} do they have in total?"
            total = sum(counts[p] for p in people)
            answer = str(total)

        elif task == "compare_yesno":
            a, b = people[0], people[1] if n_people > 1 else (people[0], people[0])
            question = f"Does {a} have more {item} than {b}?"
            answer = "yes" if counts[a] > counts[b] else "no"

        else:
            raise ValueError(f"Unknown task: {task}")

        samples.append({"context": context, "question": question, "answer": answer})

    if shuffle:
        # shuffle python list deterministically using rng permutation
        perm = rng.permutation(len(samples)).tolist()
        samples = [samples[i] for i in perm]

    return samples