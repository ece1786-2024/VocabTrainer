from agent.queryagent import QueryAgent


class VocabTrainer:
    def __init__(self):
        self.query_agent = QueryAgent()

    def run(self):
        print("Welcome to the Query Agent!")
        print("Describe your learning goal in a few sentences.")
        print("For example: 'I want to prepare for the IELTS exam' or 'I need travel vocabulary.'")
        print("Type 'exit' to quit.")

        try:
            while True:
                user_input = input("\nEnter your learning goal: ").strip()
                if user_input.lower() == "exit":
                    print("Thank you for using the Query Agent. Goodbye!")
                    break
                try:
                    result = self.query_agent.query(user_input)
                    print("\nGPT-generated analysis and word list:")
                    print(result)
                except Exception as e:
                    print(f"An error occurred: {e}")
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    trainer = VocabTrainer()
    trainer.run()
