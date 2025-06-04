# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0325-32B")
# model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0325-32B")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#olmo_model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B-Instruct", torch_dtype=torch.float16, device_map="auto")

# Function to ask a question
def ask_olmo(question):
    # Follow the instruction-tuned formatting expected by the model
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant response
    return response.split("<|assistant|>")[-1].strip()

# Example usage
question = "With no explanation or further context at all, fill in the blank with a type of room. \"I think I see a tables, chairs, computer here. Therefore, this place is most probably _____\""
answer = ask_olmo(question)
print("OLMo says:", answer)