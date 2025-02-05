import ollama
import time
import networkx as nx

# DSMG Implementation using a Dynamic Graph
class DynamicSemanticMemoryGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.timestamp = time.time()

    def add_context(self, user_message, assistant_response):
        # Add nodes for user message and assistant response
        user_node = f"user_{hash(user_message)}"
        assistant_node = f"assistant_{hash(assistant_response)}"

        self.graph.add_node(user_node, content=user_message, timestamp=self.timestamp, type="user")
        self.graph.add_node(assistant_node, content=assistant_response, timestamp=self.timestamp, type="assistant")
        
        # Add edge between user message and assistant response
        self.graph.add_edge(user_node, assistant_node, weight=1.0)
        self.timestamp += 1  # Increment timestamp for temporal ordering

    def retrieve_context(self, key_phrase):
        # Search for relevant nodes based on a key phrase
        matching_nodes = [
            (node, data)
            for node, data in self.graph.nodes(data=True)
            if key_phrase.lower() in data["content"].lower()
        ]
        # Return sorted results by timestamp for relevance
        return sorted(matching_nodes, key=lambda x: x[1]["timestamp"], reverse=True)

# Chat Function with DSMG
def chat_with_ollama_dsmg(dsmg, messages, key_phrase=None):
    if key_phrase:
        # Retrieve relevant context from DSMG
        relevant_context = dsmg.retrieve_context(key_phrase)
        context_summary = "\n".join([data["content"] for _, data in relevant_context[:5]])
        messages.insert(1, {"role": "system", "content": f"Relevant context: {context_summary}"})
    
    # Chat with the model
    response = ollama.chat(model="llama3.1:8b", messages=messages)
    dsmg.add_context(messages[-1]["content"], response["message"]["content"])
    print(response["message"]["content"])
    return response["message"]["content"]

# Initialize DSMG and chat history
dsmg = DynamicSemanticMemoryGraph()
messages = [{"role": "system", "content": "You are an AI assistant that remembers user inputs using DSMG."}]

# Simulate a long conversation to exceed context limit
for i in range(200):  # Adjust based on how much context Llama3:8b can hold
    user_message = f"Remember this number: {i}"
    messages.append({"role": "user", "content": user_message})
    assistant_response = chat_with_ollama_dsmg(dsmg, messages)
    messages.append({"role": "assistant", "content": assistant_response})

# Now ask if it remembers the first number
messages.append({"role": "user", "content": "What was the first number I gave you?"})
response = chat_with_ollama_dsmg(dsmg, messages, key_phrase="Remember this number: 0")

print("\nGPT's response:", response)
