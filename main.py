import sys
import time
import re
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextBrowser, QPushButton, QVBoxLayout,
                             QWidget, QMenu, QAction, QInputDialog, QLineEdit, QLabel,
                             QListView, QPlainTextEdit)

from nlp import (get_response, update_bot_json, load_response_data,
                 is_greeting, is_farewell)

# Load pre-trained BERT tokenizer and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad').to(device)

def generate_answer(question, context):
    max_chunk_length = 350
    stride = 150

    # Tokenize the input text
    encoded_inputs = tokenizer.encode_plus(
        question, context, return_tensors='pt', max_length=max_chunk_length, truncation=True, stride=stride,
        padding="max_length"
    )
    input_ids_windows = encoded_inputs["input_ids"].split(max_chunk_length, dim=1)
    attention_mask_windows = encoded_inputs["attention_mask"].split(max_chunk_length, dim=1)
    inputs_list = [{"input_ids": ids.to(device), "attention_mask": mask.to(device)} for ids, mask in
                   zip(input_ids_windows, attention_mask_windows)]

    best_answer = None
    best_score = -1
    best_context = None

    for inputs in inputs_list:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Find the start and end positions of the answer in the context
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        try:
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores) + 1
            answer_score = (start_scores[0][answer_start] + end_scores[0][answer_end - 1]).item()

            if answer_score > best_score:
                best_score = answer_score
                answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
                context_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                context_start = max(answer_start - 50, 0)
                context_end = min(answer_end + 50, len(context_tokens))
                best_context = tokenizer.convert_tokens_to_string(context_tokens[context_start:context_end])
                best_answer = answer
        except:
            pass

    if best_answer is None:
        return "I'm sorry, I don't know the answer to that.", context
    else:
        return best_answer, best_context

# Load the context from a file
with open("context.txt", "r") as f:
    context = f.read()

# GUI settings
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class Chatbot(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.chatbot_mode = "normal"

    def init_ui(self):
        self.setWindowTitle("Chatbot")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.label = QLabel("Welcome to PCS Assistant")
        layout.addWidget(self.label)

        self.text_area = QTextBrowser()
        layout.addWidget(self.text_area)

        self.input_field = QPlainTextEdit()
        layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send)
        layout.addWidget(self.send_button)

        central_widget.setLayout(layout)

        self.create_menu()

        self.show()

    def create_menu(self):
        menu_bar = self.menuBar()

        options_menu = QMenu("&Options", self)
        menu_bar.addMenu(options_menu)

        add_kw_resp_action = QAction("Add Keywords & Responses", self)
        add_kw_resp_action.triggered.connect(self.add_keywords_and_responses)
        options_menu.addAction(add_kw_resp_action)

        chatbot_mode_menu = QMenu("&Chatbot Mode", self)
        menu_bar.addMenu(chatbot_mode_menu)

        normal_chatbot_action = QAction("Normal Chatbot", self)
        normal_chatbot_action.triggered.connect(lambda: self.set_chatbot_mode("normal"))
        chatbot_mode_menu.addAction(normal_chatbot_action)

        bert_chatbot_action = QAction("BERT QA Chatbot", self)
        bert_chatbot_action.triggered.connect(lambda: self.set_chatbot_mode("bert_qa"))
        chatbot_mode_menu.addAction(bert_chatbot_action)

    def send(self):
        user_input = self.input_field.toPlainText()
        if user_input.strip() == '':
            return

        self.text_area.append("You -> " + user_input)
        self.input_field.clear()

        self.text_area.append("Bot -> Typing...")
        QTimer.singleShot(1000, lambda: self.type_bot_response(user_input))

    def type_bot_response(self, user_input):
        self.text_area.undo()
        if self.chatbot_mode == "normal":
            bot_response = get_response(user_input)
        else:
            bot_response, _ = generate_answer(user_input, context)
        bot_response = re.sub(r'(https?://\S+)', r'<a href="\1">\1</a>', bot_response)
        self.text_area.append("Bot -> " + bot_response)

    def set_chatbot_mode(self, mode):
        self.chatbot_mode = mode

    def add_keywords_and_responses(self):
        keyword, ok = QInputDialog.getText(self, "Add Keyword", "Enter the keyword:")
        if ok:
            response, ok = QInputDialog.getText(self, "Add Response", "Enter the response (add 'http://' or 'https://' for hyperlinks):")
            if ok:
                update_bot_json(keyword, keyword, response)
                load_response_data("bot.json")

def main():
    app = QApplication(sys.argv)
    chatbot = Chatbot()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

