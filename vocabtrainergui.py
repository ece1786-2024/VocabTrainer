import gradio as gr
from agent.queryagent import QueryAgent
from agent.rankingagent import RankingAgent
from agent.questionagent import QuestionAgent
from embedding.glove import GloveEmbedding
from agent.analyzeragent import AnalyzerAgent
from vectordb import VectorDB
import numpy as np
import os
import pickle


MAX_PROBLEM_NUM = 20
MAX_MATCHING_WORD_NUM = 10
QUERY_LOG_FILE = "query_log.pkl"

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
    
    def save_query_log(self):
        try:
            # Ensure the directory for QUERY_LOG_FILE exists
            directory = os.path.dirname(QUERY_LOG_FILE)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Save the query log to the file
            with open(QUERY_LOG_FILE, "wb") as f:
                pickle.dump(self.query_log, f)
                print(f"Query log successfully saved to: {QUERY_LOG_FILE}")

        except Exception as e:
            print(f"Failed to save query log. Error: {e}")
    
    def load_query_log(self):
        try:
            if os.path.exists(QUERY_LOG_FILE):
                with open(QUERY_LOG_FILE, "rb") as f:
                    try:
                        self.query_log = pickle.load(f)
                        print(f"Query log loaded from disk: {self.query_log}")
                    except EOFError:
                        # Handle case where the file exists but is empty or corrupted
                        self.query_log = {}
                        print(f"Query log file {QUERY_LOG_FILE} is empty. Starting with an empty log.")
            else:
                self.query_log = {}
                print("No existing query log found. Starting with an empty log.")
        except Exception as e:
            # Catch unexpected errors and initialize an empty log
            self.query_log = {}
            print(f"Error loading query log: {e}. Starting with an empty log.")

    def run(self):
        with gr.Blocks(title='VocabTrainer', theme=gr.themes.Soft()) as demo:
            component_map = {}
            components = []
            self.load_query_log()

            def start_btn_click(user_input):
                if user_input in self.query_log:
                    candidate_vocab = self.query_log[user_input]
                    print("Loaded candidate_vocab from query_log: ", candidate_vocab)
                else:
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

                    # Save query log
                    self.query_log[user_input] = candidate_vocab
                    self.save_query_log()

                # Check whether the user has already mastered the words
                vocab_table = [item for item in candidate_vocab if item[2] < 0.5]
                if len(vocab_table) == 0:
                    print("Mastered all words")
                    alert_message = "You have already mastered all the necessary words for this learning goal."
                    return [gr.update(value=alert_message, visible=True)] + [gr.update(visible=False)] * (len(components) - 1)

                selected_words = self.ranking_agent.query(vocab_table=vocab_table, num_words=self.num_words)
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
                    return f'<strong><p style="color: hsl({hue}, 100%, 20%);">Score: {score:.2f}</p></strong>'

                updates = [gr.update() for _ in range(len(components))]
                updates[component_map['quiz_submit_btn']] = gr.update(visible=False)
                updates[component_map['info']] = gr.update(visible=True)
                data = args[component_map["question-data"]]
                
                for i, question in enumerate(data["multiple-choice"]):
                    user_ans = args[component_map[f'1-{i+1}-a']]
                    user_ans = chr(question['choices'].index(user_ans) + ord('A'))
                    score_map = self.analyzer_agent.query(question, user_ans)
                    for word, rating in list(score_map.items()):
                        if self.db.update_understanding_rating(word=word, new_rating=rating) == False:
                            score_map[word] = 0
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
                    for word, rating in list(score_map.items()):
                        if self.db.update_understanding_rating(word=word, new_rating=rating) == False:
                            score_map[word] = 0
                    for j, word in enumerate(question['words']):
                        score_html = get_score_html(score_map[word])
                        score_html += f"<p><strong>Correct answer: {question['correct_matches'][f'{j+1}']}</strong></p>"
                        updates[component_map[f'2-{i+1}-{j+1}-s']] = gr.update(visible=True, value=score_html)
                
                for i, question in enumerate(data["short-answer"]):
                    user_ans = args[component_map[f'3-{i+1}-a']]
                    score_map = self.analyzer_agent.query(question, user_ans)
                    for word, rating in list(score_map.items()):
                        if self.db.update_understanding_rating(word=word, new_rating=rating) == False:
                            score_map[word] = 0
                    score_html = get_score_html(score_map[question['word']])
                    updates[component_map[f'3-{i+1}-s']] = gr.update(visible=True, value=score_html)

                for i, question in enumerate(data["scenario-based"]):
                    user_ans = args[component_map[f'4-{i+1}-a']]
                    user_ans = chr(question['choices'].index(user_ans) + ord('A'))
                    score_map = self.analyzer_agent.query(question, user_ans)
                    for word, rating in list(score_map.items()):
                        if self.db.update_understanding_rating(word=word, new_rating=rating) == False:
                            score_map[word] = 0
                    score_html = get_score_html(score_map[question['word']])
                    score_html += f"<p><strong>Correct answer: {question['correct_answer']}</strong></p>"
                    updates[component_map[f'4-{i+1}-s']] = gr.update(visible=True, value=score_html)
                
                return updates

            # Center-aligned title and matching font size/style
            gr.Markdown(
                """
                <div style="text-align: center; font-family: sans-serif; font-size: 28px; font-weight: bold;">
                    VocabTrainer
                </div>
                """,
                elem_id="learning-goals-header"
            )

            # Main Interface Layout: Two Columns
            with gr.Row(visible=True) as main_interface:
                # Left Column: Add a New Learning Goal
                with gr.Column():
                    gr.Markdown(
                        """
                        <p style="font-family: sans-serif; font-size: 16px; line-height: 1.6;">
                        To add a new learning goal, insert a few sentences. For example:
                        </p>
                        <blockquote style="font-family: sans-serif; font-size: 16px; line-height: 1.6;">
                            I want to learn words related to traveling in the IELTS word list.
                        </blockquote>
                        """,
                        elem_id="new-learning-goal"
                    )
                    user_input = gr.Textbox(label="Your Learning Goal:")
                    component_map['user_input'] = len(components)
                    components.append(user_input)
                    start_btn = gr.Button("Confirm Goal", variant="primary")
                    start_btn.click(start_btn_click, user_input, components)

                # Right Column: Previous Learning Goals
                with gr.Column():
                    gr.Markdown(
                        """
                        <p style="font-family: sans-serif; font-size: 16px; line-height: 1.6;">
                        Alternatively, choose a previous learning goal:
                        </p>
                        """,
                        elem_id="previous-goals-header"
                    )
                    
                    # Dynamically add buttons for previous learning goals
                    for goal in self.query_log.keys():  # Pass the goal (string) as input
                        goal_button = gr.Button(
                            value=goal,
                            elem_id=f"goal-button-{goal}"  # Add unique IDs for buttons
                        )
                        goal_button.click(
                            lambda g=goal: start_btn_click(g),  # Pass the learning goal (key)
                            inputs=[],
                            outputs=components
                        )

            # Custom CSS to style previous goal buttons
            gr.HTML(
                """
                <style>
                #learning-goals-header {
                    text-align: center;
                    margin-bottom: 20px;
                }
                #new-learning-goal, #previous-goals-header {
                    font-family: sans-serif;
                    font-size: 16px;
                }
                button[id^="goal-button-"] {
                    background-color: #E6F4F1;  /* Light blue background */
                    border: 1px solid #B0E0E6;  /* Light blue border */
                    color: #1E4D2B;  /* Dark green text */
                    font-weight: normal;  /* Remove bold text */
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                }
                button[id^="goal-button-"]:hover {
                    background-color: #CCE7E3;  /* Darker blue on hover */
                }
                </style>
                """
            )
            
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
            components.append(main_interface)
            component_map['ui-2'] = len(components)
            components.append(quiz_interface)
            component_map['quiz_submit_btn'] = len(components)
            components.append(quiz_submit_btn)
            component_map['info'] = len(components)
            components.append(gr.HTML('<strong><u style="color: hsl(120, 100%, 20%);">Results are saved!</u></strong>', visible=False))

            alert_component = gr.HTML(value="", visible=False)
            components.append(alert_component)
            component_map['alert'] = len(components) - 1

            start_btn.click(start_btn_click, user_input, components)
            quiz_back_btn.click(quiz_back_btn_click, None, components)
            quiz_submit_btn.click(quiz_submit_btn_click, components[:component_map[f'question-data'] + 1], components)

        demo.launch()


if __name__ == '__main__':
    trainer = VocabTrainerGUI()
    trainer.run()
