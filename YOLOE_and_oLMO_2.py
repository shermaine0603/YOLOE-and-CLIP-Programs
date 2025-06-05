from ultralytics import YOLOE
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Prompt free (YOLOE model)
model = YOLOE("yoloe-11l-seg-pf.pt")

results = model.predict(
    "Photo 5.jpg",
    show=True,
    save=True,
    show_conf=False
)

list = []
# Extract the results
for r in results:
    boxes = r.boxes.xyxy.cpu().tolist()
    cls = r.boxes.cls.cpu().tolist()
    
    # Display results
    for b, c in zip(boxes, cls):
        print(f"box: {b}, cls: {model.names[int(c)]}")
        list.append(model.names[int(c)])

# for i in list:
#     print("There is a", i)

#print(list)

string = ', a '.join(list)
print(string)

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
question = f"Fill in the blank with a type of room, with NO explanation or further context at all. I think I see {string} here. Therefore, this place is most probably _____"
# print(question)
answer = ask_olmo(question)
print("OLMo says:", answer)
