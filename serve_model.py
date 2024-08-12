from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

app = FastAPI()

# Load the Wizard Coder model and tokenizer
model_name = "microsoft/phi-2"  # Replace with the actual model identifier
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.post("/review", response_class=PlainTextResponse)
async def review_code(request: Request):
    data = await request.json()
    code_snippet = data['code']
    
    # Define a prompt to guide the model to generate a code review comment
    prompt = (f"Please provide a detailed code review comment for the following code snippet:\n"
              f"{code_snippet}\n"
              f"Include suggestions for improvements, potential issues, and any other relevant feedback.")
    
    # Tokenize the input code and generate a review comment
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=250, num_return_sequences=1)
    
    # Decode the generated output and format it
    review_comment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Optionally, clean up the response
    review_comment = review_comment.strip()

    return PlainTextResponse(review_comment)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
