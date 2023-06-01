import pandas as pd
import flan_t5

# Path to the FAQ text file
faq_file_path = "/path/to/faq.txt"

# Read the FAQ text file into a Pandas DataFrame
faq_df = pd.read_csv(faq_file_path, delimiter="\n", header=None)

# Create a Flan-T5 model
model = flan_t5.FlanT5()

# Load the FAQ text file into the Flan-T5 model
model.load_data(faq_df)

# Start a Flan-T5 interactive session
session = model.start_session()

# Get a question from the user
question = input("What is your question? ")

# Ask Flan-T5 to answer the question
answer = session.ask(question)

# Print the answer to the user
print(answer)
