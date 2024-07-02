import torch 
import streamlit as st
from transformers import pipeline, RobertaForQuestionAnswering, RobertaTokenizer

# Load pre-trained RoBERTa model and tokenizer
model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")

# Define a function to perform question answering
def answer_question(question, context):
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

# Streamlit app
def main():
    st.title("RoBERTa Question Answering")
    question = st.text_input("Enter your question:")
    context = st.text_area("Enter the context:")
    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question.")
        elif not context:
            st.warning("Please enter the context.")
        else:
            answer = answer_question(question, context)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()