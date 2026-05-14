from kb_loader import get_all_knowledge

kb = get_all_knowledge()

def simple_search(query):
    results = []

    query = query.lower()

    for category, items in kb.items():
        for item in items:
            text = str(item).lower()

            if query in text:
                results.append(item)

    return results