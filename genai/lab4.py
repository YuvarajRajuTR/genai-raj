from py2neo import Graph
 
# Connect to the local Neo4j instance
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "neo4jtest"))
 
def query_graph():
    query = """
    MATCH (p:Person)-[:WORKED_WITH]->(p2:Person)
    RETURN p.name AS person, p.profession AS profession, p2.name AS collaborator
    """
    result = graph.run(query).data()
    return result
 
result = query_graph()

import requests

# Use Ollama to generate a detailed response based on the retrieved answer
ollama_url = "http://localhost:11434/api/generate"

# Prepare a prompt from the graph result
prompt = "Generate a biography based on the following information:\n\n"
for record in result:
    print(f"{record['person']} is a {record['profession']} who worked with {record['collaborator']}.\n")
 
