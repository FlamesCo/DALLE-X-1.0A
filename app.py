from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
model = AutoModelWithLMHead.from_pretrained("bigscience/bloom") 
model.to(device)
model.eval()

# Choose the text to encode (can be a long one!)
text ="input" 


# Encode the text and get the prediction scores for each token
encoded_text = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length') 
input_ids = encoded_text['input_ids'].to(device)  
attention_mask = encoded_text['attention_mask'].to(device)  
outputs = model(input_ids, attention_mask=attention_mask)  

predictions = outputs[0]
