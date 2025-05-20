import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import spacy
from transformers import BertTokenizer, BertForSequenceClassification

class PoliticalTopicModeller(object):

    #------------------------------------------------------------------------#
    

    def __init__(self):
        print("Initializing tokenizer and model...")
        self.model = AutoModelForSequenceClassification.from_pretrained("manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        try: 
            # Check if GPU is available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            which_device = torch.cuda.get_device_name(0)
            print(f"Using device: {which_device}")

            # Move the model to GPU
            self.model = self.model.to(self.device)
        except Exception as e:
            print(e)

    #------------------------------------------------------------------------#

    def get_year(self,file_path):

        path = Path(file_path)
        file_name = path.name

        match = re.search(r'(19|20)\d{2}', file_name)
        if match:
            year = int(match.group(0))
            print("Extracted year:", year)
        else:
            print("No year found in filename.")
            year = np.nan

        self.year = year

        return self.year

    #------------------------------------------------------------------------#

    def get_candidate(self, file_path):

        path = Path(file_path)
        parent_folder = path.parent

        self.current_candidate = parent_folder.name
        return self.current_candidate

    #------------------------------------------------------------------------#

    def read_text_file(self, file_path):
        """Read text from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    #------------------------------------------------------------------------#

    def segment_by_sentences(self, text):
        """Split text into sentences."""

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text.strip())
        
        sentences = []
        buffer = ""

        for sent in doc.sents:
            raw_text = sent.text
            stripped_text = raw_text.strip()

            if not stripped_text:
                continue  # skip empty sentences

            if len(stripped_text) < 100:
                if buffer:
                    buffer += " " + stripped_text
                else:
                    buffer = stripped_text
            else:
                if buffer:
                    combined = buffer + " " + stripped_text
                    sentences.append(combined.strip())
                    buffer = ""
                else:
                    sentences.append(stripped_text)

        # If there's anything left in the buffer at the end, add it
        if buffer:
            sentences.append(buffer.strip())

        print(f'nr of sentences: {len(sentences)}')
        return sentences

    #------------------------------------------------------------------------#

    def identify_topics(self, sentences):
        

        topic_dict = defaultdict(lambda: defaultdict(list))

        for idx, sentence in enumerate(sentences):

            if ((idx == 0) or (idx == len(sentences) - 1)):
                context = sentence

            else:
                prec_sent = sentences[idx - 1]
                foll_sent = sentences[idx + 1]
                context = " ".join([prec_sent,foll_sent])

            

            inputs = self.tokenizer(sentence,
                    context,
                    return_tensors="pt",
                    max_length=300, 
                    padding="max_length",
                    truncation=True
                    )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            logits = self.model(**inputs).logits

            # Calculate probabilities
            probabilities = torch.softmax(logits, dim=1).tolist()[0]
            probabilities_dict = {self.model.config.id2label[index]: round(probability * 100, 2) 
                            for index, probability in enumerate(probabilities)}
            sorted_probabilities = dict(sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True))

            predicted_class = self.model.config.id2label[logits.argmax().item()]

            # Get the similarity score (probability) for the predicted class
            similarity_score = probabilities_dict[predicted_class]
            
            # Store the sentence, topic, and similarity score
            topic_dict[self.current_candidate][self.year].append({
                'sentence': sentence,
                'topic': predicted_class,
                'similarity_score': similarity_score
            })

        return topic_dict

    
    #------------------------------------------------------------------------#

    def split_text_by_topics(self, topics_dictionary):
        grouped_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"text": "", "avg_similarity": [], "count": 0})))

        for candidate, years in topics_dictionary.items():
            for year, entries in years.items():
                for entry in entries:
                    topic = entry['topic']
                    sentence = entry['sentence']
                    similarity = entry['similarity_score']
                    
                    # Add the sentence to the text
                    grouped_dict[candidate][year][topic]["text"] += sentence.strip() + " "
                    
                    # Store the similarity score for later averaging
                    grouped_dict[candidate][year][topic]["avg_similarity"].append(similarity)
                    
                    # Increment count
                    grouped_dict[candidate][year][topic]["count"] += 1
        
        # Calculate average similarity scores
        for candidate, years in grouped_dict.items():
            for year, topics in years.items():
                for topic, data in topics.items():
                    if data["avg_similarity"]:
                        data["avg_similarity"] = sum(data["avg_similarity"]) / len(data["avg_similarity"])
                    else:
                        data["avg_similarity"] = 0.0
        
        # Convert to dictionary format
        result = {}
        for candidate in grouped_dict:
            result[candidate] = {}
            for year in grouped_dict[candidate]:
                result[candidate][year] = {}
                for topic in grouped_dict[candidate][year]:
                    result[candidate][year][topic] = {
                        "text": grouped_dict[candidate][year][topic]["text"],
                        "avg_similarity": grouped_dict[candidate][year][topic]["avg_similarity"],
                        "sentence_count": grouped_dict[candidate][year][topic]["count"]
                    }
        
        return result



#------------------------------------------------------------------------#

    def topic_text_by_output(self, output):

        results = []
        
        for candidate, years in output.items():
            print(f"\nCandidate: {candidate}")
            
            for year, topics in years.items():
                
                for topic, text in topics.items():
                    
                    results.append((topic, text))  # collect each pair

        return results

    #------------------------------------------------------------------------#

    def run_topic_modelling(self, file_path):

        # Read text
        text = self.read_text_file(file_path)

        candidate = self.get_candidate(file_path)

        year = self.get_year(file_path)

        # Segment into sentences
        sentences = self.segment_by_sentences(text)

        # Identify topics
        topics_dict = self.identify_topics(sentences)

        output = self.split_text_by_topics(topics_dict)

        return output


#------------------------------------------------------------------------#

class PersonalityModeller(object):

    #------------------------------------------------------------------------#

    def __init__(self, max_length = 512):

        print("Initializing tokenizer and model...")

        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
        self.model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")

        try: 
            # Check if GPU is available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            which_device = torch.cuda.get_device_name(0)
            print(f"Using device: {which_device}")

            # Move the model to GPU
            self.model = self.model.to(self.device)
        except Exception as e:
            print(e)

        return

    #------------------------------------------------------------------------#

    def split_by_tokens(self, text):
        
        #print("Tokenizing text...")
        tokens = self.tokenizer.encode(text, add_special_tokens=True)


        chunks = [tokens[i:i+self.max_length] for i in range(0, len(tokens), self.max_length)]
        print(f"Number of chunks to process: {len(chunks)} (each with up to {self.max_length} tokens)")


        return chunks

    #------------------------------------------------------------------------#

    def predict_personality(self, text):
        
        chunks = self.split_by_tokens(text)

        all_predictions = []

        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx+1}/{len(chunks)}...")
            try:
                chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
                encoded = self.tokenizer(chunk_text, truncation=True, padding=True, return_tensors="pt", max_length=self.max_length)
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.no_grad():
                    outputs = self.model(**encoded)
                logits = outputs.logits.squeeze().cpu().numpy()
                all_predictions.append(logits)
            except Exception as e:
                print(f"Failed to process chunk {idx+1}")
                raise e

        return all_predictions

    #------------------------------------------------------------------------#
    
    def calculate_bold(self, row, Neuro_est = -0.05 , Extra_est = 0.23,Openn_est = 0.03 , Agree_est = -0.28 , Consc_est = 0.16):
        '''
        narcicism level from https://onlinelibrary.wiley.com/doi/full/10.1002/pmh.1262?casa_token=d_q0tChNaOAAAAAA%3APyRAOKGRG453BMyeOZh7WbTfRlViMM2uO9O-SSLcb3J1cw_ADk-1rk5aKzjsKtB5ktW22SXR7mJYIoE
        Bold correlated with: 
        Neuroticism (−0.13), 
        Extraversion (0.30), 
        Openness (0.13), 
        Agreeableness (−0.24)
        Conscientiousness (0.21)
        '''
        Neuroticism = row['Neuroticism'] 
        Extraversion = row['Extroversion'] 
        Openness = row['Openness'] 
        Agreeableness = row['Agreeableness']
        Conscientiousness = row['Conscientiousness'] 

        narci = Neuroticism * Neuro_est + Extraversion * Extra_est + Openness * Openn_est + Agreeableness * Agree_est + Conscientiousness * Consc_est

        return narci

    #------------------------------------------------------------------------#

    def get_results(self, text):

        all_predictions = self.predict_personality(text)
        #print(f'this is all predictions: {all_predictions}')
        
        label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

        trait_dicts = [dict(zip(label_names, lst)) for lst in all_predictions]
        
        return trait_dicts

#------------------------------------------------------------------------#






