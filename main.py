import os 

from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Configuración de CORS
origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = dotenv_values(".env")

os.environ["OPENAI_API_KEY"] = config["API_KEY"]

# Modelo de datos para recibir la petición
class CompletionRequest(BaseModel):
    prompt: str
    
# Crea una instancia de memoria y de la cadena de conversación
memoria = ConversationBufferMemory()
llm = ChatOpenAI()
chatbot = ConversationChain(llm=llm, memory=memoria, verbose=True)

@app.post("/completions")
async def completions(request: CompletionRequest):
    try:
        # Obtiene la respuesta de la IA
        response = chatbot.predict(input=request.prompt)

        # Devuelve la respuesta de la IA
        return {"response": response}
    except Exception as e:
        # Manejo de excepciones si algo sale mal
        raise HTTPException(status_code=500, detail=str(e))