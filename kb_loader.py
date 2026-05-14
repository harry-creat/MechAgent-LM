import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


materials = load_json("knowledge_base/materials.json")
structures = load_json("knowledge_base/structures.json")
supports = load_json("knowledge_base/supports.json")


def get_all_knowledge():
    return {
        "materials": materials,
        "structures": structures,
        "supports": supports
    }