import gradio as gr
from agent.queryagent import QueryAgent
from agent.rankingagent import RankingAgent
from agent.questionagent import QuestionAgent
from embedding.glove import GloveEmbedding
from agent.analyzeragent import AnalyzerAgent
from vectordb import VectorDB
import numpy as np


MAX_PROBLEM_NUM = 20
MAX_MATCHING_WORD_NUM = 10

class VocabTrainerGUI():
    def __init__(self):
        self.query_agent = QueryAgent()
        self.ranking_agent = RankingAgent()
        self.question_agent = QuestionAgent()
        self.analyzer_agent = AnalyzerAgent()
        self.embedding = GloveEmbedding()
        self.db = VectorDB()
        self.num_words = 7
        self.num_questions = 10

    def run(self):
        with gr.Blocks(title='VocabTrainer', theme=gr.themes.Soft()) as demo:
            component_map = {}
            components = []

            def start_btn_click(user_input):
                user_query = self.query_agent.query(user_input)
                print('user_query:', user_query) # returns exam, topic, and keywords
                user_query_exam = user_query['exam']

                # Compute the average of keywords
                candidate_vocab = []
                for word in user_query['keywords']:
                    keyword_emb = self.embedding.encode(word)
                    keyword_table = self.db.query_by_similarity(keyword_emb, n_results=10)
                    for row in keyword_table:
                        if user_query_exam == None or (user_query_exam == "GRE" and row['GRE'] == True) or (user_query_exam == "IELTS" and row['IELTS'] == True):
                            candidate_vocab.append((row['word'], row['CEFR'], row['understanding_rating']))
                print("candidate_vocab: ", candidate_vocab)

                # Check whether the user has already mastered the words
                if len(candidate_vocab) == 0:
                    print("You have already mastered all the necessary words!")

                selected_words = self.ranking_agent.query(vocab_table=candidate_vocab, num_words=self.num_words)
                print('selected_words:', selected_words)
                
                print("Generating questions...")
                data = self.question_agent.query(word_list=selected_words, num_questions=self.num_questions)
                print(data)

                updates = [gr.update() for _ in range(len(components))]
                updates[component_map['question-data']] = data
                updates[component_map['ui-1']] = gr.update(visible=False)
                updates[component_map['ui-2']] = gr.update(visible=True)
                updates[component_map['quiz_submit_btn']] = gr.update(visible=True)
                
                # Hide all questions initially
                for i in range(component_map['1-1-q'], component_map[f'4-{MAX_PROBLEM_NUM}-s'] + 1):
                    updates[i] = gr.update(visible=False)
                
                # Populate multiple-choice questions
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
                
                # Populate matching questions
                for i, question in enumerate(data["matching"]):
                    q = f"### {i+1}. Match the words with the correct definitions.\n\n"
                    q += f"**Definitions:**\n\n"
                    definitions = question['definitions']
                    definition_labels = [chr(ord('A') + i) for i in range(len(definitions))]
                    for label, definition in zip(definition_labels, definitions):
                        q += f"{label}. {definition}\n\n"
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
                
                # Populate short answer questions
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
                
                # Populate scenario-based questions
                print("DEBUG-start-btn: Scenario-based questions data:", data["scenario-based"])

                for i, question in enumerate(data["scenario-based"]):
                    updates[component_map[f'4-{i+1}-q']] = gr.update(
                        visible=True,
                        value=f"### {i+1}. {question['question']}"
                    )
                    updates[component_map[f'4-{i+1}-a']] = gr.update(
                        visible=True,
                        interactive=True,
                        choices=question['choices'],
                        value=None
                    )

                return updates

            def quiz_back_btn_click():
                updates = [gr.update() for _ in range(len(components))]
                updates[component_map['info']] = gr.update(visible=False)
                updates[component_map['ui-2']] = gr.update(visible=False)
                updates[component_map['ui-1']] = gr.update(visible=True)
                return updates
            
            def quiz_submit_btn_click(*args):
                def get_score_html(score):
                    hue = score * 120
                    return f'<strong><p style="color: hsl({hue}, 100%, 50%);">Score: {score:.2f}</p></strong>'

                updates = [gr.update() for _ in range(len(components))]
                updates[component_map['quiz_submit_btn']] = gr.update(visible=False)
                updates[component_map['info']] = gr.update(visible=True)
                data = args[component_map["question-data"]]
                
                for i, question in enumerate(data["multiple-choice"]):
                    user_ans = args[component_map[f'1-{i+1}-a']]
                    user_ans = chr(question['choices'].index(user_ans) + ord('A'))
                    score_map = self.analyzer_agent.query(question, user_ans)
                    for word, rating in score_map.items():
                        self.db.update_understanding_rating(word=word, new_rating=rating)
                    score_html = get_score_html(score_map[question['word']])
                    score_html += f"<p><strong>Correct answer: {question['correct_answer']}</strong></p>"
                    updates[component_map[f'1-{i+1}-s']] = gr.update(visible=True, value=score_html)
                
                for i, question in enumerate(data["matching"]):
                    user_ans = ""
                    for j in range(len(question['words'])):
                        if j > 0:
                            user_ans += ", "
                        arg = args[component_map[f'2-{i+1}-{j+1}-a']]
                        user_ans += f'{j+1}-{arg}'
                    score_map = self.analyzer_agent.query(question, user_ans)
                    for word, rating in score_map.items():
                        self.db.update_understanding_rating(word=word, new_rating=rating)
                    for j, word in enumerate(question['words']):
                        score_html = get_score_html(score_map[word])
                        score_html += f"<p><strong>Correct answer: {question['correct_matches'][f'{j+1}']}</strong></p>"
                        updates[component_map[f'2-{i+1}-{j+1}-s']] = gr.update(visible=True, value=score_html)
                
                for i, question in enumerate(data["short-answer"]):
                    user_ans = args[component_map[f'3-{i+1}-a']]
                    score_map = self.analyzer_agent.query(question, user_ans)
                    for word, rating in score_map.items():
                        self.db.update_understanding_rating(word=word, new_rating=rating)
                    score_html = get_score_html(score_map[question['word']])
                    updates[component_map[f'3-{i+1}-s']] = gr.update(visible=True, value=score_html)

                print("DEBUG: Scenario-based questions data:", data["scenario-based"])

                for i, question in enumerate(data["scenario-based"]):
                    user_ans = args[component_map[f'4-{i+1}-a']]
                    user_ans = chr(question['choices'].index(user_ans) + ord('A'))
                    score_map = self.analyzer_agent.query(question, user_ans)
                    for word, rating in score_map.items():
                        self.db.update_understanding_rating(word=word, new_rating=rating)
                    score_html = get_score_html(score_map[question['word']])
                    score_html += f"<p><strong>Correct answer: {question['correct_answer']}</strong></p>"
                    updates[component_map[f'4-{i+1}-s']] = gr.update(visible=True, value=score_html)
                
                return updates

            with gr.Column(visible=True) as user_requirements_interface:
                gr.Markdown("# User Requirements Interface\n\n**Describe your learning goal in a few sentences. For example:**\n\n> I want to learn words related to traveling in the IELTS word list.")
                user_input = gr.Textbox(label="Your requirements:")
                component_map['user_input'] = len(components)
                components.append(user_input)
                start_btn = gr.Button("Next", variant="primary")

            with gr.Column(visible=False) as quiz_interface:
                gr.Markdown("# Vocabulary Quiz\n\n## Multiple Choice")
                for i in range(MAX_PROBLEM_NUM):
                    component_map[f'1-{i+1}-q'] = len(components)
                    components.append(gr.Markdown())
                    component_map[f'1-{i+1}-a'] = len(components)
                    components.append(gr.Radio(label="Your answer:", type="value"))
                    component_map[f'1-{i+1}-s'] = len(components)
                    components.append(gr.HTML())

                gr.Markdown("---\n\n## Matching")
                for i in range(MAX_PROBLEM_NUM):
                    component_map[f'2-{i+1}-q'] = len(components)
                    components.append(gr.Markdown())
                    for j in range(MAX_MATCHING_WORD_NUM):
                        component_map[f'2-{i+1}-{j+1}-a'] = len(components)
                        components.append(gr.Radio(type="value"))
                        component_map[f'2-{i+1}-{j+1}-s'] = len(components)
                        components.append(gr.HTML())

                gr.Markdown("---\n\n## Short Answer")
                for i in range(MAX_PROBLEM_NUM):
                    component_map[f'3-{i+1}-q'] = len(components)
                    components.append(gr.Markdown())
                    component_map[f'3-{i+1}-a'] = len(components)
                    components.append(gr.Textbox(label="Your answer:"))
                    component_map[f'3-{i+1}-s'] = len(components)
                    components.append(gr.HTML())

                gr.Markdown("---\n\n## Scenario-Based")
                for i in range(MAX_PROBLEM_NUM):
                    component_map[f'4-{i+1}-q'] = len(components)
                    components.append(gr.Markdown())
                    component_map[f'4-{i+1}-a'] = len(components)
                    components.append(gr.Radio(label="Your answer:", type="value"))
                    component_map[f'4-{i+1}-s'] = len(components)
                    components.append(gr.HTML())
                
                quiz_submit_btn = gr.Button("Submit", variant="primary")
                quiz_back_btn = gr.Button("Back", variant="secondary")

            component_map['question-data'] = len(components)
            components.append(gr.State())
            component_map['ui-1'] = len(components)
            components.append(user_requirements_interface)
            component_map['ui-2'] = len(components)
            components.append(quiz_interface)
            component_map['quiz_submit_btn'] = len(components)
            components.append(quiz_submit_btn)
            component_map['info'] = len(components)
            components.append(gr.HTML('<strong><u style="color: hsl(120, 100%, 50%);">Results are saved!</u></strong>', visible=False))

            start_btn.click(start_btn_click, user_input, components)
            quiz_back_btn.click(quiz_back_btn_click, None, components)
            quiz_submit_btn.click(quiz_submit_btn_click, components[:component_map[f'question-data'] + 1], components)

        demo.launch()


if __name__ == '__main__':
    trainer = VocabTrainerGUI()
    trainer.run()
