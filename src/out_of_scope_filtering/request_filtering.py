from sentence_transformers import SentenceTransformer, util
import spacy

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_lg")


def split_text_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def filter_sentences(request):
    sentences = split_text_into_sentences(request)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    max_tokens = 512
    request_chunks = [request[i:i + max_tokens] for i in range(0, len(request), max_tokens)]

    chunk_embeddings = model.encode(request_chunks, convert_to_tensor=True)

    similarities = []
    for sentence_embedding in sentence_embeddings:
        chunk_similarities = util.pytorch_cos_sim(sentence_embedding, chunk_embeddings)
        max_similarity = chunk_similarities.max().item()
        similarities.append(max_similarity)

    threshold = 0.5
    filtered_sentences = [sentence for sentence, similarity in zip(sentences, similarities) if similarity > threshold]

    return '. '.join(filtered_sentences)