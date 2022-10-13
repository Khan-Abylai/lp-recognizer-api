import os
import cv2
import logging
import requests
import telegram
from config import TOKEN
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import shortuuid
BASE_DIR = os.getcwd()

def start(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I am ImageToCaption and CaptionToImage bot. Please send me the image or caption. I will send you caption for image and image for caption. CaptionToImage on development stage.",
    )

def _help(update, context):
    help_msg = """
    *How to use:*
    Send a photo to the bot directly.Then we will send you caption for your photo. If you want to generate image for you caption, please send by the \captionToImage command
    """
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        parse_mode=telegram.ParseMode.MARKDOWN,
        text=help_msg
    )

def unknown(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Sorry, I didn't understand that command.",
    )

def callback_fromImage_toCaption(update, context):
    img = None
    img_file = context.bot.get_file(update.message.photo[-1].file_id)
    # TODO get caption and translate caption to this language
    # print(update.message)

    filename = os.path.join('static', 'tmp', shortuuid.uuid() + '.png')
    print(img_file)
    if img_file is not None:
        img_file.download(filename)  # temporarily dump image to file and read as OpenCV frame
        try:
            img = cv2.imread(filename, 1)
        except Exception as E:
            img = cv2.cv2.imread(filename, 1)
            print(E)
    file = cv2.imencode(".png", img)[1].tobytes ()
    print(img)
    data = {
        "image":file
    }
    url = "http://parking:9001/api/image"

    r = requests.post(url, files=data)
    print(r)
    if r.json().get("status") == True:
        label = r.json().get("label")
        prob = r.json().get("prob")
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Result for your image: label {label[0]} , probability: {prob[0]}",
        )
    else:
        logging.info("For image didnt generated caption")
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="No Result",
        )


def main():
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # handlers
    start_handler = CommandHandler("start", start)
    help_handler = CommandHandler("help", _help)
    unknown_handler = MessageHandler(Filters.command, unknown)
    ImageToCaptionHandler = MessageHandler(Filters.photo | Filters.reply | Filters.sticker, callback_fromImage_toCaption)
    # dispatchers
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(help_handler)
    dispatcher.add_handler(unknown_handler)
    dispatcher.add_handler(ImageToCaptionHandler)


    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()