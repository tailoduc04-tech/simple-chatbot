import uvicorn
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from chatbot_services import ChatbotServices

# --- Khởi tạo ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("Vui lòng thiết lập biến môi trường TELEGRAM_TOKEN")

chatbot = ChatbotServices()
application = Application.builder().token(TELEGRAM_TOKEN).build()

# --- Context Manager cho Lifespan của FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo application của bot khi server FastAPI bắt đầu
    await application.initialize()
    print("Bot application initialized.")
    
    yield  # Ứng dụng sẽ chạy ở đây
    
    # Dọn dẹp application khi server FastAPI tắt
    await application.shutdown()
    print("Bot application shutdown.")

# --- Tạo ứng dụng FastAPI với lifespan ---
app = FastAPI(lifespan=lifespan)

# --- Các hàm xử lý của Telegram ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = update.message.chat_id
    conversation = [{"role": "user", "content": user_message}]
    
    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(
        None, chatbot.ask, user_message, conversation
    )
    
    await context.bot.send_message(chat_id=chat_id, text=answer)

application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# --- Endpoints của FastAPI ---
@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return Response(status_code=200)
    except Exception as e:
        print(f"Lỗi khi xử lý webhook: {e}")
        return Response(status_code=500)

@app.get("/set_webhook")
async def set_webhook(url: str):
    webhook_url = f"{url.rstrip('/')}/webhook"
    if await application.bot.set_webhook(webhook_url):
        return {"status": f"webhook đã được thiết lập thành công tới {webhook_url}"}
    else:
        return {"status": "thiết lập webhook thất bại"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)