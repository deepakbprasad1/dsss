from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from transformers import pipeline
import logging

# Replace with your Telegram bot token
API_TOKEN = "7707424980:AAGYUUuVcuCYjTp_2bMQfFNNS8zv1cOPafY"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load TinyLlama model
logger.info("Loading TinyLlama model...")
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
logger.info("TinyLlama model loaded successfully.")

# Command handler for /start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        f"Hello! I'm your AI assistant, created by Deepak Bindhu Prasad (Student ID: 23292926, IdM: iq93aqyh).\n\n"
        "Send me a message, and I'll respond using TinyLlama!"
    )

# Message handler for processing user input
async def process(update: Update, context: CallbackContext) -> None:
    try:
        user_message = update.message.text
        logger.info(f"User message: {user_message}")

        # Add a prompt to guide the model for better responses
        prompt = f"Q: {user_message}\nA:"

        # Generate a response using TinyLlama
        response = pipe(
            prompt,
            max_length=150,  # Increase max_length for more detailed responses
            num_return_sequences=1,  # Generate only one response
            temperature=0.7,  # Adjust creativity (0 = deterministic, 1 = creative)
            top_p=0.9,  # Use top-p sampling for better quality
            truncation=True,  # Truncate input if it's too long
            pad_token_id=pipe.tokenizer.eos_token_id,  # Use EOS token for padding
        )[0]['generated_text']

        # Extract only the assistant's response (remove the prompt)
        assistant_reply = response.split("A:")[-1].strip()
        logger.info(f"Generated response: {assistant_reply}")

        # Send the response back to the user
        await update.message.reply_text(assistant_reply)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        await update.message.reply_text("Sorry, I encountered an error while generating a response.")

# Main function to start the bot
def main() -> None:
    application = Application.builder().token(API_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process))

    # Start the bot
    logger.info("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()