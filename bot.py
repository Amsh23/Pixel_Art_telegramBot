import logging
import os
from collections import defaultdict, deque
from dataclasses import replace
from typing import Dict

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import TOKEN
from image_tools import (
    ProcessingOptions,
    GENERATION_MODES,
    QUALITY_MODES,
    STYLE_ALIASES,
    STYLE_NAMES,
    cleanup_temp,
    create_comparison,
    create_palette_preview,
    get_image_info,
    process_image,
)

DOWNLOAD_DIR = "downloads"
OUTPUT_DIR = "outputs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

USER_SETTINGS: Dict[int, ProcessingOptions] = defaultdict(ProcessingOptions)
USER_HISTORY: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
LAST_IMAGE: Dict[int, str] = {}
LAST_OUTPUT: Dict[int, str] = {}

COMMANDS = [
    ("start", "Start the Pixel Art bot"), ("help", "Show help"), ("styles", "List available pixel styles"),
    ("settings", "Show your current settings"), ("pixel", "Use Classic OpenCV mode"), ("ai_pixel", "Use AI Pixel Art mode"), ("hd_pixel", "Use HD Pixel style"),
    ("gameboy", "Use GameBoy style"), ("nes", "Use NES style"), ("snes", "Use SNES style"),
    ("minecraft", "Use Minecraft style"), ("retro", "Use Retro RPG style"), ("lego", "Use LEGO style"), ("pico8", "Use Pico-8 style"),
    ("c64", "Use Commodore64 style"), ("palette", "Preview current palette"), ("compare", "Create before/after comparison"),
    ("info", "Show last image information"), ("history", "Show your last 10 images"), ("about", "About this bot"),
]


def settings_text(options: ProcessingOptions) -> str:
    return (
        f"Style: {STYLE_NAMES.get(options.style, options.style)}\n"
        f"Mode: {options.generation_mode.upper()}\n"
        f"Quality: {options.quality.upper()}\n"
        f"Pixel Size: {options.pixel_size or 'Auto'}\nPalette Size: {options.palette_size}\n"
        f"Outline: {options.outline.title()}\nDithering: {'On' if options.dithering else 'Off'}\n"
        f"Sharpen: {'On' if options.sharpen else 'Off'}\nDenoise: {'On' if options.denoise else 'Off'}"
    )


def settings_keyboard(options: ProcessingOptions) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Mode", callback_data="menu:mode"), InlineKeyboardButton("🎨 Style", callback_data="menu:style")],
        [InlineKeyboardButton("⚙️ Quality", callback_data="menu:quality"), InlineKeyboardButton("🤖 AI Prompt", callback_data="action:ai_prompt")],
        [InlineKeyboardButton("🔲 Pixel Size", callback_data="menu:pixel"), InlineKeyboardButton("🌈 Palette Size", callback_data="menu:palette")],
        [InlineKeyboardButton("✒️ Outline", callback_data="menu:outline"), InlineKeyboardButton("✨ Effects", callback_data="menu:effects")],
        [InlineKeyboardButton("✅ Process Image", callback_data="action:process")],
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎨 Pixel Art Bot\n\nSend an image, then choose style, pixel size, palette, outline, and dithering.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Commands:\n" + "\n".join(f"/{name} - {desc}" for name, desc in COMMANDS))


async def styles(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Available styles:\n" + "\n".join(f"• {name}" for name in STYLE_NAMES.values()))


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await update.message.reply_text(settings_text(USER_SETTINGS[uid]), reply_markup=settings_keyboard(USER_SETTINGS[uid]))


async def set_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    parts = update.message.text.split(maxsplit=1)
    cmd = parts[0].lstrip("/").lower()
    opts = USER_SETTINGS[uid]
    if cmd == "ai_pixel":
        prompt = parts[1] if len(parts) > 1 else opts.ai_prompt
        USER_SETTINGS[uid] = replace(opts, generation_mode="ai", quality="hd", outline="thin", dithering=True, sharpen=True, ai_prompt=prompt)
        await update.message.reply_text("AI Pixel Art mode enabled. Send a photo; optional custom prompt: /ai_pixel heroic RPG knight")
        return
    style = STYLE_ALIASES[cmd]
    quality = "hd" if cmd in ("hd", "hd_pixel") else opts.quality
    generation_mode = "classic"
    palette_size = 4 if cmd == "gameboy" else (32 if cmd == "snes" else opts.palette_size)
    outline = "thin" if cmd in ("hd", "hd_pixel", "retro") else opts.outline
    USER_SETTINGS[uid] = replace(opts, style=style, quality=quality, generation_mode=generation_mode, palette_size=palette_size, outline=outline)
    await update.message.reply_text(f"Style set to {STYLE_NAMES[style]}. Send a photo or use /settings.")


async def palette(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    path = os.path.join(OUTPUT_DIR, f"palette_{uid}.png")
    create_palette_preview(USER_SETTINGS[uid].style, USER_SETTINGS[uid].palette_size, path)
    with open(path, "rb") as img:
        await update.message.reply_photo(img, caption="Current palette preview")


async def compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in LAST_IMAGE or uid not in LAST_OUTPUT:
        await update.message.reply_text("Send and process an image first.")
        return
    path = os.path.join(OUTPUT_DIR, f"compare_{uid}.png")
    create_comparison(LAST_IMAGE[uid], LAST_OUTPUT[uid], path)
    with open(path, "rb") as img:
        await update.message.reply_photo(img, caption="Before / After")


async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in LAST_IMAGE:
        await update.message.reply_text("No image yet. Send a photo first.")
        return
    details = get_image_info(LAST_IMAGE[uid])
    await update.message.reply_text("Image information:\n" + "\n".join(f"{k}: {v}" for k, v in details.items()))


async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    items = USER_HISTORY[update.effective_user.id]
    if not items:
        await update.message.reply_text("No processed images yet.")
        return
    await update.message.reply_text("Last images:\n" + "\n".join(f"{i+1}. {item}" for i, item in enumerate(items)))


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Pixel Art Bot creates professional classic OpenCV pixel art and AI-ready/Stable-Diffusion pixel art using OpenCV, NumPy, Pillow, and python-telegram-bot v21+.")


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cleanup_temp([DOWNLOAD_DIR, OUTPUT_DIR])
    uid = update.effective_user.id
    photo_obj = update.message.photo[-1]
    file = await context.bot.get_file(photo_obj.file_id)
    input_path = os.path.join(DOWNLOAD_DIR, f"{uid}_{photo_obj.file_unique_id}.jpg")
    await file.download_to_drive(input_path)
    LAST_IMAGE[uid] = input_path
    await update.message.reply_text("Image received. Choose Classic OpenCV or AI Pixel settings, then tap Process Image.", reply_markup=settings_keyboard(USER_SETTINGS[uid]))


async def process_last_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in LAST_IMAGE:
        await update.effective_message.reply_text("Send a photo first.")
        return
    output_path = os.path.join(OUTPUT_DIR, f"{uid}_{os.path.basename(LAST_IMAGE[uid])}.png")
    await update.effective_message.reply_text("🖼 Processing pixel art...")
    process_image(LAST_IMAGE[uid], output_path, USER_SETTINGS[uid])
    LAST_OUTPUT[uid] = output_path
    pixel_label = USER_SETTINGS[uid].pixel_size or "Auto"
    USER_HISTORY[uid].appendleft(f"{STYLE_NAMES[USER_SETTINGS[uid].style]} - {pixel_label}px - {os.path.basename(output_path)}")
    with open(output_path, "rb") as img:
        await update.effective_message.reply_photo(img, caption="✅ Done! Use /compare, /palette, or /info for more.")


async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    data = query.data
    opts = USER_SETTINGS[uid]
    if data == "menu:mode":
        await query.edit_message_text("Choose generation mode:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(v.title(), callback_data=f"set:generation_mode:{v}") for v in GENERATION_MODES]])); return
    if data == "menu:style":
        rows = [[InlineKeyboardButton(name, callback_data=f"set:style:{key}")] for key, name in STYLE_NAMES.items()]
        await query.edit_message_text("Choose style:", reply_markup=InlineKeyboardMarkup(rows)); return
    if data == "menu:quality":
        await query.edit_message_text("Choose quality mode:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(v.upper(), callback_data=f"set:quality:{v}") for v in QUALITY_MODES]])); return
    if data == "menu:pixel":
        await query.edit_message_text("Choose pixel size:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Auto", callback_data="set:pixel_size:auto"), InlineKeyboardButton("4", callback_data="set:pixel_size:4"), InlineKeyboardButton("8", callback_data="set:pixel_size:8")], [InlineKeyboardButton(str(v), callback_data=f"set:pixel_size:{v}") for v in (16,32,64)]])); return
    if data == "menu:palette":
        await query.edit_message_text("Choose palette size:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(str(v), callback_data=f"set:palette_size:{v}") for v in (8,16,32)], [InlineKeyboardButton(str(v), callback_data=f"set:palette_size:{v}") for v in (64,128,256)]])); return
    if data == "menu:outline":
        await query.edit_message_text("Choose outline:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(v.title(), callback_data=f"set:outline:{v}") for v in ("off","thin","medium","thick")]])); return
    if data == "menu:effects":
        await query.edit_message_text("Toggle effects:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(f"Dither: {'On' if opts.dithering else 'Off'}", callback_data="set:dithering:toggle")], [InlineKeyboardButton(f"Sharpen: {'On' if opts.sharpen else 'Off'}", callback_data="set:sharpen:toggle")], [InlineKeyboardButton(f"Denoise: {'On' if opts.denoise else 'Off'}", callback_data="set:denoise:toggle")]])); return
    if data == "action:ai_prompt":
        await query.edit_message_text("Send /ai_pixel followed by your custom prompt, for example:\n/ai_pixel cute RPG wizard sprite, blue cloak, idle pose")
        return
    if data == "action:process":
        await process_last_image(update, context); return
    _, field, value = data.split(":", 2)
    if field == "dithering":
        opts = replace(opts, dithering=not opts.dithering)
    elif field == "sharpen":
        opts = replace(opts, sharpen=not opts.sharpen)
    elif field == "denoise":
        opts = replace(opts, denoise=not opts.denoise)
    elif field == "quality":
        opts = replace(opts, quality=value)
    elif field == "generation_mode":
        opts = replace(opts, generation_mode=value)
    elif field == "style":
        opts = replace(opts, style=value)
    elif field == "pixel_size":
        opts = replace(opts, pixel_size=None if value == "auto" else int(value))
    elif field == "palette_size":
        opts = replace(opts, palette_size=int(value))
    elif field == "outline":
        opts = replace(opts, outline=value)
    USER_SETTINGS[uid] = opts
    await query.edit_message_text("Settings updated:\n" + settings_text(opts), reply_markup=settings_keyboard(opts))


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled Telegram bot error", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("Sorry, something went wrong while processing your request.")


async def post_init(app: Application):
    await app.bot.set_my_commands([BotCommand(name, desc) for name, desc in COMMANDS])


def main():
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    for cmd in ("start",): app.add_handler(CommandHandler(cmd, start))
    app.add_handler(CommandHandler("help", help_command)); app.add_handler(CommandHandler("styles", styles)); app.add_handler(CommandHandler("settings", settings))
    for cmd in ("pixel", "ai_pixel", "hd", "hd_pixel", "gameboy", "nes", "snes", "minecraft", "retro", "lego", "pico8", "c64"):
        app.add_handler(CommandHandler(cmd, set_style))
    app.add_handler(CommandHandler("palette", palette)); app.add_handler(CommandHandler("compare", compare)); app.add_handler(CommandHandler("info", info)); app.add_handler(CommandHandler("history", history)); app.add_handler(CommandHandler("about", about))
    app.add_handler(CallbackQueryHandler(callbacks)); app.add_handler(MessageHandler(filters.PHOTO, photo)); app.add_error_handler(error_handler)
    logger.info("Bot Started")
    app.run_polling()


if __name__ == "__main__":
    main()
