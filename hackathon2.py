import requests
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

# Step 1: Fetch the dataset from the online JSON file
def fetch_dataset(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch dataset")

# URL of the hosted JSON dataset
dataset_url = 'https://raw.githubusercontent.com/Arimakouse/Hackathon-repository/refs/heads/main/constitution_qa.json'  # Replace with your actual URL
data = fetch_dataset(dataset_url)

# Prepare questions and answers
questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

# Step 2: Prepare the data for training
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
X = tokenizer.texts_to_sequences(questions)
X = pad_sequences(X)

# Encode answers as indices
y = np.arange(len(answers))  # Simple encoding for demonstration

# Step 3: Build the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64))
model.add(LSTM(64))
model.add(Dense(len(answers), activation='softmax'))  # Output layer for multiple classes

# Step 4: Compile and train the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# Step 5: Save the model and tokenizer
model.save('legal_chatbot_model.h5')
import pickle
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Step 6: Create the Kivy GUI
class LegalChatbotApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.input_box = TextInput(hint_text='What would you like to know?', multiline=False)
        self.submit_button = Button(text='Submit')
        self.response_label = Label(text='')

        self.submit_button.bind(on_press=self.on_submit)

        self.layout.add_widget(self.input_box)
        self.layout.add_widget(self.submit_button)
        self.layout.add_widget(self.response_label)

        return self.layout

    def on_submit(self, instance):
        user_input = self.input_box.text.strip()
        if user_input == 'exit':
            self.response_label.text = "Thank you for using the Legal Chatbot. Goodbye!"
            return
        
        # Predict the response
        response = predict_response(user_input)
        self.response_label.text = response

def predict_response(user_input):
    # Load the trained model and tokenizer
    model = load_model('legal_chatbot_model.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Preprocess the input
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=model.input_shape[1])
    prediction = model.predict(padded_sequence)
    response_index = np.argmax(prediction)

    # Return the corresponding answer
    return answers[response_index]

if __name__ == '__main__':
    LegalChatbotApp().run()