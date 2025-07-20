import pickle
from typing import Final
import os
from dotenv import load_dotenv

from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from telegram import Update
from telegram.ext import ContextTypes, Application, filters, CommandHandler, MessageHandler

load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
BOT_USERNAME: Final = '@anti_cyberbullyingbot'

# Load model and tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
model = load_model('model.h5')


# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello, this is a Deep learning model for textual classification of cyberbullying')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('''
    /start - Starts conversations
/help - Shows this message again
/about - Shows important information about the bot
    ''')


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("This project focuses on using the CNN-GRU model to detect cyberbullying, "
                                    "showcasing an impressive accuracy of up to 95%. Its exceptional accuracy allows "
                                    "for precise identification of harmful online behavior, differentiating it from "
                                    "normal communication effectively. Compared to other models, this approach stands "
                                    "out due to its superior accuracy, ensuring timely intervention to prevent "
                                    "further harm. Moreover, the model's versatility enables it to handle diverse "
                                    "datasets from various online platforms, making it a powerful tool in combating "
                                    "cyberbullying across different formats and communication channels'")


def predict_cyberbullying(text, tokenizer):
    sequences = tokenizer.texts_to_sequences([text])
    print('Tokenized Sequences:', sequences)

    max_sequence_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequences)[0][0]
    print('Prediction:', prediction)

    return prediction


# This store user warnings
user_warnings = {}


# Modify the handle_message function
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type = update.message.chat.type
    text = update.message.text

    print(f'User ({update.message.from_user.id}) in {message_type}: "{text}"')

    if 'tokenizer' not in context.chat_data:
        context.chat_data['tokenizer'] = loaded_tokenizer  # Store the tokenizer for both group and private chat

    # Tokenize and predict in a single step
    tokenizer = context.chat_data['tokenizer']
    prediction = predict_cyberbullying(text, tokenizer)

    # Print the prediction value before responding
    print('Prediction in handle_message:', prediction)

    # Adjust the decision threshold as needed
    threshold = 0.5  # You can experiment with different threshold values
    is_cyberbullying = prediction >= threshold
    print('Is Cyberbullying:', is_cyberbullying)

    user_id = update.message.from_user.id

    # Check if the user has any warnings
    if user_id not in user_warnings:
        user_warnings[user_id] = 0

    # Respond only if cyberbullying is detected
    if is_cyberbullying:
        user_warnings[user_id] += 1
        warnings_left = 3 - user_warnings[user_id]

        response = f"Warning: Cyberbullying detected! You have {warnings_left} warnings left."
        print('Bot:', response)
        await update.message.reply_text(response)

        # If the user reaches the maximum number of warnings, remove them from the group
        if user_warnings[user_id] >= 3:
            response = "You have reached the maximum number of warnings. You are being removed from the group."
            print('Bot:', response)
            await update.message.reply_text(response)

            # Remove the user from the group
            try:
                await context.bot.ban_chat_member(update.message.chat.id, user_id)
            except Exception as e:
                print(f"Error kicking user: {e}")

            # Reset the user's warnings after being removed
            user_warnings[user_id] = 0

    return  # Add this line to exit the function after responding


def handle_response(prediction) -> str:
    # Model is binary classification (0 or 1)
    is_cyberbullying = prediction >= 0.5
    result = "Cyberbullying Detected!" if is_cyberbullying else "No Cyberbullying Detected"

    return result


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


if __name__ == '__main__':
    print('Starting bot....')
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('about', about_command))

    # Message
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Error
    app.add_error_handler(error)

    print('Polling ...')
    app.run_polling(poll_interval=3)
