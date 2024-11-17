import openai


openai.api_key = 'put-your-key-here'

def analyze_context_and_generate_words(user_input):
    """
    Use GPT to analyze the user's input, determine the goal, and generate relevant words.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a vocabulary assistant. Given a user's input, identify whether their goal is exam preparation "
                "(e.g., IELTS, GRE) or learning a specific area (e.g., travel, academic research). Generate a list of "
                "10 relevant words for their context. Include each word's difficulty level (A1 to C2)."
            ),
        },
        {
            "role": "user",
            "content": f"User's input: '{user_input}'. Analyze the goal and generate a list of words with difficulty levels.",
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300,
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"].strip()

def query_agent():
    """
    Query Agent that interacts with the user to generate relevant vocabulary words based on their learning goal.
    """
    print("Welcome to the Query Agent!")
    print("Describe your learning goal in a few sentences.")
    print("For example: 'I want to prepare for the IELTS exam' or 'I need travel vocabulary.'")
    print("Type 'exit' to quit.")

    while True:
        # Get user input
        user_input = input("\nEnter your learning goal: ").strip()
        if user_input.lower() == "exit":
            print("Thank you for using the Query Agent. Goodbye!")
            break

        try:
            # Analyze context and generate words
            result = analyze_context_and_generate_words(user_input)
            print("\nGPT-generated analysis and word list:")
            print(result)
        except Exception as e:
            print(f"An error occurred: {e}")

query_agent()
