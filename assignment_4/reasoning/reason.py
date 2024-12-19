import neo4j
import os
from ollama import Client



PROJECT_DIR = "assignment_4"
OLLAMA_URI = 'http://localhost:11434'
LLM_MODEL = "llama3.1:70b"
DATABASE_NAME = "neo4j"
DATABASE_URI = "bolt://localhost:7687"
DATABASE_USERNAME = "neo4j"
DATABASE_PASSWORD = "juhujuhu"



def get_db_schema(driver: neo4j.Driver, database: str) -> tuple[str, str]:
    """
    Extracts node schema and relation schema from the graph databse. These two schema
    can then be used to prompt the LLM.
    :param driver: The Neo4j driver that is used to interface with the database
    :param database: Name of the database
    :return: A tuple of strings (node_schema, rel_schema).
    """

    # the code that is provided in this function is there to help you get started
    # if you have different approach of achieving the goal for this exercise, go ahead!
    node_schema = ""
    relation_schema = ""


    query = "CALL apoc.meta.schema()"
    with driver.session(database=database) as session:
        result = session.run(query)

        # schema_data["value"] will contain a lot of information about relationships and nodes
        schema_data = result.data()[0]

        # if you want to provide additional information like all possible categories,
        # you could run an additional cypher query like the following
        # res = session.run("MATCH (n:Node) RETURN DISTINCT n.category;")

    # set to be filled in parsing to prevent duplicate relations
    unique_relations = set()

    # parse schema_data to build a readable schema of the graph database
    for label, details in schema_data['value'].items():
        # ignore relation entries as they don't ontain information on direction or start/target nodes
        if details['type'] == 'node':
            # build node_schema
            node_schema += f"NodeType: {label}, Properties: {", ".join(f"{key}: {value['type']}" for key, value in details['properties'].items())}\n"

            # fill unique_relations from nodes' "relationships" value
            for rel_type, rel_details in details.get('relationships', {}).items():
                start_node = label
                direction = rel_details.get('direction', "unknown")
                properties = ", ".join(
                    f"{k}: {v['type']}" for k, v in rel_details.get('properties', {}).items()) or "None"

                # implement direction of relation and add it to unique_relations
                for target_node in rel_details['labels']:
                    if direction == 'out':
                        relation = f"({start_node})-[{rel_type}]->({target_node}), Properties: {properties}"
                    elif direction == 'in':
                        relation = f"({target_node})-[{rel_type}]->({start_node}), Properties: {properties}"
                    else:
                        continue
                    unique_relations.add(relation)

            # build relation_schema from unique_relations
            relation_schema = "\n".join(unique_relations)

    return node_schema, relation_schema


def ask_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Given a model, system prompt, and user prompt, returns the LLM's response.
    :param model: The Ollama model that should be used
    :param system_prompt: The system prompt for Ollama
    :param user_prompt: The user prompt for Ollama
    :return: The model response as a string. Do NOT return the response dict!
    """

    # NOTE: if you want to run your code locally and only use a remote machine for the LLM,
    # use SSH port forwarding for port 11434 (the default Ollama port)
    # so something like `ssh -L 11434:localhost:11434 rzname@mmcXZY.informatik.uni-augsburg.de`
    # this way, you don't have to run the Ollama server locally. The only requirement
    # on you local machine is the ollama python package.

    client = Client(host=OLLAMA_URI)
    response = client.chat(model='qwen2:7b', messages=[
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': user_prompt
        }
    ])

    return response["message"]["content"]


def extract_cypher_query(raw_response: str) -> str:
    """
    The output from `ask_llm` will contain more than just the required output data.
    This function extracts only the relevant part. You can solve this task using a regex
    or basic string comparison.
    :param raw_response: The raw response from ask_llm()
    :return: The cleaned response
    """
    raise NotImplementedError()


def create_system_prompt() -> str:
    # read prompt template
    template_file = open(os.path.join(PROJECT_DIR, "reasoning/prompt.txt"))
    template = template_file.read()
    template_file.close()

    # get database schema from neo4j graph database
    driver = neo4j.GraphDatabase.driver(DATABASE_URI, auth=(DATABASE_USERNAME, DATABASE_PASSWORD))
    node_schema, rel_schema = get_db_schema(driver, DATABASE_NAME)
    driver.close()

    system_prompt = template.replace("{{node_schema}}", node_schema).replace("{{rel_schema}}", rel_schema)

    return system_prompt





def main():
    """
    Put your code for exercise 4.1g) here.
    Either implement a command line interface or run all 4 required prompts here.
    """

    system_prompt = create_system_prompt()

    print(ask_llm(LLM_MODEL, system_prompt, "This is a test, generate me a sample cypher query. Choose whatever you want"))


if __name__ == "__main__":
    main()
