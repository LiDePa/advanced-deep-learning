import neo4j
import os
from ollama import Client
import json
import re


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    # send system and user prompt to the LLM and retain response
    client = Client(host=OLLAMA_URI)
    response = client.chat(model=LLM_MODEL, messages=[
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

    # find first { and las } in raw response and define them as the json_block
    start = raw_response.find("{")
    end = raw_response.rfind("}") + 1
    json_block = raw_response[start:end]

    # get only the query from json_block
    data_response = json.loads(json_block)
    query = data_response.get("query", "")

    # check for restricted keywords before returning query
    restricted_keywords = ["delete", "merge", "insert"]
    if any(keyword in query.lower() for keyword in restricted_keywords):
        return ""
    else:
        return query


# read system prompt template and insert database schema
def create_system_prompt(driver: neo4j.Driver) -> str:
    # read prompt template
    with open(os.path.join(SCRIPT_DIR, "prompt.txt")) as template_file:
        template = template_file.read()

    # get database schema from neo4j graph database
    node_schema, rel_schema = get_db_schema(driver, DATABASE_NAME)

    # insert database schema into system prompt template
    system_prompt = template.replace("{{node_schema}}", node_schema).replace("{{rel_schema}}", rel_schema)

    return system_prompt


def ask_database(driver: neo4j.Driver, database: str, user_prompt: str, system_prompt: str):
    # send prompt to model and retain the produced query
    raw_response = ask_llm(LLM_MODEL, system_prompt, user_prompt)
    query = extract_cypher_query(raw_response)
    print(query)

    # run query on the database
    with driver.session(database=database) as session:
        result = session.run(query).data()

    return result




def main():
    """
    Put your code for exercise 4.1g) here.
    Either implement a command line interface or run all 4 required prompts here.
    """
    driver = neo4j.GraphDatabase.driver(DATABASE_URI, auth=(DATABASE_USERNAME, DATABASE_PASSWORD))

    system_prompt = create_system_prompt(driver)

    user_prompts = ["How many actors starred in ”Joe Versus the Volcano”?",
                    "When was the director of the movie ”Stand By Me” born?",
                    "Which actors starred in both ”The Matrix Revolutions” and ”The Matrix Reloaded”?",
                    "Who is the youngest director in the dataset?"]

    # user_prompts = ["How many people on the image are in the air?",
    #                 "Which team is in possession of the ball?",
    #                 "Is any player trying to block the attack?",
    #                 "How many people are watching the ball?"]

    for user_prompt in user_prompts:
        result = ask_database(driver, DATABASE_NAME, user_prompt, system_prompt)
        print(result)

    driver.close()


if __name__ == "__main__":
    main()


# First try:
#
# MATCH (m:Movie {title: "Joe Versus the Volcano"})-[:ACTED_IN]-(p:Person) RETURN COUNT(p)
# [{'COUNT(p)': 3}]
# MATCH (m:Movie {title: 'Stand By Me'})-[:DIRECTED]-(p:Person) RETURN p.born
# [{'p.born': 1947}]
# MATCH (p1:Person)-[:ACTED_IN]->(m1:Movie {title: "The Matrix Revolutions"}), (p2:Person)-[:ACTED_IN]->(m2:Movie {title: "The Matrix Reloaded"}) WHERE p1 = p2 RETURN DISTINCT p1.name
# [{'p1.name': 'Keanu Reeves'}, {'p1.name': 'Carrie-Anne Moss'}, {'p1.name': 'Laurence Fishburne'}, {'p1.name': 'Hugo Weaving'}]
# MATCH (p:Person)-[:DIRECTED]->(:Movie) RETURN p ORDER BY p.born ASC LIMIT 1
# [{'p': {'born': 1930, 'name': 'Clint Eastwood'}}]

# Second Try (First try after adapting prompt.txt):
#
# MATCH (m:Movie {title: "Joe Versus the Volcano"})-[:ACTED_IN]-(a) RETURN COUNT(a)
# [{'COUNT(a)': 3}]
# MATCH (p:Person)-[:DIRECTED]->(m:Movie {title: "Stand By Me"}) RETURN p.born
# [{'p.born': 1947}]
# MATCH (p1:Person)-[:ACTED_IN]->(:Movie {title: "The Matrix Revolutions"}) MATCH (p2:Person)-[:ACTED_IN]->(:Movie {title: "The Matrix Reloaded"}) WHERE p1 = p2 RETURN DISTINCT p1.name
# [{'p1.name': 'Keanu Reeves'}, {'p1.name': 'Carrie-Anne Moss'}, {'p1.name': 'Laurence Fishburne'}, {'p1.name': 'Hugo Weaving'}]
# MATCH (p:Person)-[:DIRECTED]->(m:Movie) RETURN p ORDER BY p.born DESC LIMIT 1
# [{'p': {'born': 1967, 'name': 'Andy Wachowski'}}]