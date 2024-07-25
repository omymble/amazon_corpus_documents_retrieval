from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Check if MPS (Metal Performance Shaders) is available
device = torch.device("mps") if torch.has_mps else torch.device("cpu")

# Load the model and tokenizer
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

text = """
I've been doing a lot of reading on early Christianity, say AD 30-200. I have learned that there were A LOT of itinerant apocalyptic type preachers in that day. Many people were referred to as "Son of Man". People ascribed powers to lots of wandering magicians and mystics, and "raising people from the dead" was not as uncommon as it seems to be today. So the question that I am looking to answer, (extra-biblically, not from a faith perspective) is, "Why Jesus?" Why not Appolonius of Tyana, or any of many other preacher types that proposed and preached in a similar fashion. Was it Paul's zeal? James'? The fact that Christianity demanded monotheism and other gods didn't mind if you held to previous divine allegiances?
Can anyone recommend some books that tackle this question, and compare Jesus' ministry directly with his contemporaries?
Thanks for any books you can point out.....
"""

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
