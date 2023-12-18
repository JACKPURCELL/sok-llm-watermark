import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the tokenizer and model for both teacher and student models
teacher_model_name = "gpt3-model-name"  # Replace with the actual model name
student_model_name = "llama-2-model-name"  # Replace with the actual model name

teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)

student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)

# Example input text
input_texts = ["Example input 1", "Example input 2"]

# Prepare the dataset
teacher_outputs = []
for input_text in input_texts:
    inputs = teacher_tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = teacher_model(**inputs)
    teacher_outputs.append(outputs.logits)

# Distillation process
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-5)

for input_text, teacher_output in zip(input_texts, teacher_outputs):
    inputs = student_tokenizer(input_text, return_tensors="pt")
    labels = torch.argmax(teacher_output, dim=-1)

    # Forward pass
    student_output = student_model(**inputs).logits

    # Compute loss
    loss = criterion(student_output.view(-1, student_output.size(-1)), labels.view(-1))

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")

# Save the fine-tuned student model
student_model.save_pretrained("fine_tuned_llama_2")

