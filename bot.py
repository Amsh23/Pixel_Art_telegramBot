import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from image_tools import pixel_art
from config import TOKEN


os.makedirs("downloads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "🎨 Pixel Art Bot\n\n"
        "فقط یک عکس ارسال کن."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "/start\n"
        "/help\n\n"
        "یک عکس بفرست تا به Pixel Art تبدیل شود."
    )


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    photo = update.message.photo[-1]

    file = await context.bot.get_file(photo.file_id)

    input_path = f"downloads/{photo.file_unique_id}.jpg"

    output_path = f"outputs/{photo.file_unique_id}.png"

    await file.download_to_drive(input_path)

    await update.message.reply_text("🖼 در حال ساخت Pixel Art ...")

    pixel_art(
        input_path,
        output_path,
        pixel_size=16
    )

    with open(output_path, "rb") as img:

        await update.message.reply_photo(
            photo=img,
            caption="✅ Done!"
        )


def main():

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo
        )
    )

    print("Bot Started")

    app.run_polling()


if __name__ == "__main__":
    main()