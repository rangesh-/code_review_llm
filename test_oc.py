import ollama

# Function to chat with Ollama
def chat_with_ollama(messages):
    response = ollama.chat(model="llama3.1:8b", messages=messages)
    print(response['message']['content'])
    return response['message']['content']

# Initialize chat history
messages = [{"role": "system", "content": "You are an AI assistant that remembers user inputs."}]

# Simulate a long conversation to exceed context limit
for i in range(200):  # Adjust based on how much context Llama3:8b can hold
    messages.append({"role": "user", "content": f"Remember this number: {i}"})
    messages.append({"role": "assistant", "content": chat_with_ollama(messages)})

# Now ask if it remembers the first number
messages.append({"role": "user", "content": "What was the first number I gave you?"})
response = chat_with_ollama(messages)

print("\nGPT's response:", response)
