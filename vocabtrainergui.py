from agent.queryagent import QueryAgent
from agent.rankingagent import RankingAgent
from agent.questionagent import QuestionAgent
from embedding.glove import GloveEmbedding
from quiz import Quiz
from vectordb import VectorDB
import gradio as gr
import json
import numpy as np


MAX_PROBLEM_NUM = 20
MAX_MATCHING_WORD_NUM = 10

class VocabTrainerGUI():
    def __init__(self):
        self.query_agent = QueryAgent()
        self.ranking_agent = RankingAgent()
        self.question_agent = QuestionAgent()
        self.embedding = GloveEmbedding()
        self.db = VectorDB()
        self.num_words = 5
        self.num_questions = 5

    def run(self):
        with gr.Blocks(title='VocabTrainer', theme=gr.themes.Soft()) as demo:
            component_map = {}
            components = []

            def start_btn_click(user_input):
                user_query = self.query_agent.query(user_input)
                print('user_query:', user_query) # returns exam, topic, and keywords

                # Compute the average of keywords
                query_vector = np.zeros(self.embedding.dim)
                query_actual_len = 0
                for word in user_query['keywords']:
                    if self.embedding.contains(word):
                        query_actual_len += 1
                        query_vector += self.embedding.encode(word)
                query_vector /= query_actual_len
                
                # Query words similar to the keywords
                candidate_table = self.db.query_by_similarity(query_vector, n_results=200)
                candidate_vocab = []
                for row in candidate_table:
                    if row['understanding_rating'] < 0.5:
                        candidate_vocab.append((row['word'], row['CEFR'], row['understanding_rating']))
                
                # Check whether the user has already mastered the words
                if len(candidate_vocab) == 0:
                    print("You have already mastered all the necessary words!")

                selected_words = self.ranking_agent.query(vocab_table=candidate_vocab, num_words=self.num_words)
                print('selected_words:', selected_words)
                
                print("Generating questions...")
                data_json = self.question_agent.query(word_list=selected_words, num_questions=self.num_questions)
                print(data_json)
                data = json.loads(data_json)
                updates = [gr.update() for _ in range(len(components))]
                updates[component_map['user_input']] = gr.update(value="")
                updates[component_map['ui-1']] = gr.update(visible=False)
                updates[component_map['ui-2']] = gr.update(visible=True)
                for i in range(component_map['1-1-q'], component_map[f'3-{MAX_PROBLEM_NUM}-a'] + 1):
                    updates[i] = gr.update(visible=False)
                for i, question in enumerate(data["multiple-choice"]):
                    updates[component_map[f'1-{i+1}-q']] = gr.update(
                        visible=True,
                        value=f"### {i+1}. {question['question']}"
                    )
                    updates[component_map[f'1-{i+1}-a']] = gr.update(
                        visible=True,
                        interactive=True,
                        choices=question['choices'],
                        value=None
                    )
                for i, question in enumerate(data["matching"]):
                    q = f"### {i+1}. Match the words with the correct definitions.\n\n"
                    q += f"**Words:**\n\n"
                    for word in question['words']:
                        q += f"+ {word}\n"
                    q += f"\n**Definitions:**\n\n"
                    definitions = question['definitions']
                    definition_labels = [chr(ord('A') + i) for i in range(len(definitions))]
                    for label, definition in zip(definition_labels, definitions):
                        q += f"+ {label}. {definition}\n"
                    updates[component_map[f'2-{i+1}-q']] = gr.update(
                        visible=True,
                        value=q
                    )
                    for j, word in enumerate(question['words']):
                        updates[component_map[f'2-{i+1}-{j+1}-a']] = gr.update(
                            label=f'Choose a definition for "{word}":',
                            visible=True,
                            interactive=True,
                            choices=definition_labels,
                            value=None
                        )
                for i, question in enumerate(data["short-answer"]):
                    updates[component_map[f'3-{i+1}-q']] = gr.update(
                        visible=True,
                        value=f"### {i+1}. {question['question']}"
                    )
                    updates[component_map[f'3-{i+1}-a']] = gr.update(
                        visible=True,
                        interactive=True,
                        value=""
                    )
                return updates


            with gr.Column(visible=True) as user_requirements_interface:
                gr.Markdown("# User Requirements Interface")
                user_input = gr.Textbox(label="Your requirements:")
                component_map['user_input'] = len(components)
                components.append(user_input)
                start_btn = gr.Button("Next", variant="primary")

            with gr.Column(visible=False) as quiz_interface:
                gr.Markdown("## Multiple Choice")
                for i in range(MAX_PROBLEM_NUM):
                    component_map[f'1-{i+1}-q'] = len(components)
                    components.append(gr.Markdown())
                    component_map[f'1-{i+1}-a'] = len(components)
                    components.append(gr.Radio(label="Your answer:", type="value"))
                gr.Markdown("---")
                gr.Markdown("## Matching")
                for i in range(MAX_PROBLEM_NUM):
                    component_map[f'2-{i+1}-q'] = len(components)
                    components.append(gr.Markdown())
                    for j in range(MAX_MATCHING_WORD_NUM):
                        component_map[f'2-{i+1}-{j+1}-a'] = len(components)
                        components.append(gr.Radio(type="value"))
                gr.Markdown("---")
                gr.Markdown("## Short Answer")
                for i in range(MAX_PROBLEM_NUM):
                    component_map[f'3-{i+1}-q'] = len(components)
                    components.append(gr.Markdown())
                    component_map[f'3-{i+1}-a'] = len(components)
                    components.append(gr.Textbox(label="Your answer:"))
                quiz_submit_btn = gr.Button("Submit", variant="primary")
            
            component_map['ui-1'] = len(components)
            components.append(user_requirements_interface)
            component_map['ui-2'] = len(components)
            components.append(quiz_interface)

            start_btn.click(start_btn_click, user_input, components)

        demo.launch()


if __name__ == '__main__':
    trainer = VocabTrainerGUI()
    trainer.run()