from sklearn.metrics import classification_report


def evaluate_model(true_aspects, pred_aspects, true_sentiments, pred_sentiments):
    aspect_report = classification_report(true_aspects, pred_aspects, output_dict=True)
    sentiment_report = classification_report(true_sentiments, pred_sentiments, output_dict=True)
    return aspect_report, sentiment_report


def evaluate_model_aspects(true_aspects, pred_aspects):
    aspect_report = classification_report(true_aspects, pred_aspects, output_dict=True)
    # sentiment_report = classification_report(true_sentiments, pred_sentiments, output_dict=True)
    return aspect_report
