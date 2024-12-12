import json
from argparse import ArgumentParser


def _make_neo4j_nodename(raw_label: str):
    return raw_label.replace(" ", "_").replace("-", "_").capitalize()


def cli():
    parser = ArgumentParser(description="Converts annotation file to Cypher query. This query can then be used to populate the Neo4j database.")
    parser.add_argument("json", help="Input path for the JSON annotation data from the annotation tool")
    parser.add_argument("out", help="Output path for the generated Cypher query")
    args = parser.parse_args()

    with open(args.json) as f:
        anno = json.load(f)

    uid2name = {}

    with open(args.out, "w") as f:
        f.write("// clear old data\nMATCH (n) DETACH DELETE n;\n\n")
        f.write("// nodes\n")
        for instance in anno["instances"]:
            uid = instance["uid"]
            lbls = [_make_neo4j_nodename(l) for l in instance["labels"]]
            assert lbls[0]
            labels_str = ":".join(lbls)
            node_name = f"{lbls[0]}_{uid}"
            uid2name[uid] = node_name
            props = instance.get("attributes", {})
            props["uid"] = uid
            attr_str = ", ".join(
                (
                    f"{k.replace(' ','_')}:'{v}'"
                    for (k, v) in instance.get("attributes", {}).items()
                )
            )
            f.write(f"CREATE ({node_name}:{labels_str} {{{attr_str}}})\n")
        f.write("\n")
        f.write("// relations\n")
        for rel in anno["relations"]:
            sbj = uid2name[rel["sbj"]]
            obj = uid2name[rel["obj"]]
            cat = rel["rel"]
            cat: str = cat.upper().replace(" ", "_")
            f.write(f"CREATE ({sbj})-[:{cat}]->({obj})\n")


if __name__ == "__main__":
    cli()
