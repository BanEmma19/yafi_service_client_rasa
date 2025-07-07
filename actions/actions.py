from rasa_sdk import Action
from rasa_sdk.events import UserUtteranceReverted
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


class ActionKeywordSimilarityFallback(Action):
    def name(self):
        return "action_keyword_similarity_fallback"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")

        with open("data/nlu.json", encoding='utf-8') as f:
            nlu_data = json.load(f)

        # Extraire tous les exemples avec leurs intents
        examples = []
        intents = []
        for intent in nlu_data['nlu']:
            for ex in intent.get('examples', '').split("\n"):
                example = ex.strip().lstrip("-").strip()
                if example:
                    examples.append(example)
                    intents.append(intent['intent'])

        # Calculer similarité
        vectorizer = TfidfVectorizer().fit_transform(examples + [user_message])
        cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
        best_match_index = cosine_similarities.argmax()
        best_intent = intents[best_match_index]
        similarity_score = cosine_similarities[best_match_index]

        if similarity_score > 0.4:
            dispatcher.utter_message(text=f"Je ne suis pas sûr, mais vous parlez peut-être de : *{best_intent}* ?")
        else:
            dispatcher.utter_message(text="Je n'ai pas compris votre demande, pouvez-vous reformuler ?")

        return [UserUtteranceReverted()]
