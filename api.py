from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import uvicorn

app = FastAPI(title="AI Reader Backend")


# Описываем, что мы ждем от фронтенда (читалки)
class SummarizeRequest(BaseModel):
    text: str
    chunk_size: int = 3000  # Макс символов в одном куске (для примера)


# Вспомогательная функция для запроса к Ollama
def ask_ollama(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:14b",
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m"  # Держим в памяти 5 минут, т.к. кусков будет несколько подряд
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['response']
    return ""


@app.post("/api/summarize")
async def summarize_text(req: SummarizeRequest):
    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Текст пустой")

    print(f"📥 Получен текст длиной {len(req.text)} символов.")

    # 1. ЧАНКОВАНИЕ (пока примитивное, рубим по количеству символов)
    # В идеале резать по абзацам (\n\n), чтобы не рвать предложения посередине
    chunks = [req.text[i:i + req.chunk_size] for i in range(0, len(req.text), req.chunk_size)]
    print(f"✂️ Текст разбит на {len(chunks)} частей.")

    chunk_summaries = []

    # 2. Мэтап MAP: Делаем выжимку каждой части
    for idx, chunk in enumerate(chunks):
        print(f"🔄 Обработка части {idx + 1}/{len(chunks)}...")
        prompt = f"Напиши краткий пересказ этого отрывка текста:\n\n{chunk}"
        summary = ask_ollama(prompt)
        chunk_summaries.append(summary)

    # 3. Этап REDUCE: Собираем финальное саммари
    if len(chunks) == 1:
        # Если текст был короткий и кусок всего один
        final_summary = chunk_summaries[0]
    else:
        print("🛠️ Сборка финального саммари из полученных кусков...")
        combined_text = "\n\n---\n\n".join(chunk_summaries)
        final_prompt = f"Объедини эти короткие пересказы в один связный и логичный текст сюжета:\n\n{combined_text}"
        final_summary = ask_ollama(final_prompt)

    print("✅ Готово!")
    return {
        "final_summary": final_summary,
        "chunks_processed": len(chunks)
    }


# Точка входа для запуска сервера
if __name__ == "__main__":
    # Запускаем веб-сервер на порту 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)