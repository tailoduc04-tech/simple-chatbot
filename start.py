from chatbot_services import initialize_chatbot, run_cli_chatbot

if __name__ == "__main__":
    app = initialize_chatbot()
    run_cli_chatbot(app)