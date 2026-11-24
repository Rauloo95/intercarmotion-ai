import os
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Cargar variables de entorno (.env)
load_dotenv()

app = FastAPI(title="InterCarMotion AI Service")

# 2. Configurar CORS para permitir peticiones desde tu frontend en React (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Inicializar cliente del LLM
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 4. Definir DTOs (Entrada y Salida estructurada)
class SearchRequest(BaseModel):
    user_query: str = Field(..., example="Busco un coche deportivo alemán por menos de 35000 euros")

class CarFilterResponse(BaseModel):
    marca: Optional[str] = Field(None, description="Marca del coche si se menciona, o null")
    tipo_combustible: Optional[str] = Field(None, description="Gasolina, Diésel, Híbrido o Eléctrico")
    presupuesto_maximo: Optional[int] = Field(None, description="Precio máximo en euros si se indica")
    kilometros_maximos: Optional[int] = Field(None, description="Kilometraje máximo si se indica")
    mensaje_asistente: str = Field(..., description="Respuesta amable y breve explicando qué búsqueda se ha aplicado")

# 5. Endpoint principal
@app.post("/api/ai/parse-search", response_model=CarFilterResponse)
async def parse_car_search(request: SearchRequest):
    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")

    system_prompt = (
        "Eres el asistente experto de InterCarMotion, un portal de compraventa de vehículos. "
        "Tu trabajo es extraer parámetros de búsqueda estructurados a partir de lo que pide el usuario. "
        "Si un dato no se menciona, déjalo en null. Sé preciso con los números."
    )

    modelos_a_probar = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    ultimo_error = None

    for modelo in modelos_a_probar:
        try:
            response = client.models.generate_content(
                model=modelo,
                contents=request.user_query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=CarFilterResponse,
                ),
            )
            return CarFilterResponse.model_validate_json(response.text)
        except Exception as e:
            ultimo_error = e
            continue  # Si da 503 o falla, pasa al siguiente modelo

    raise HTTPException(
        status_code=503,
        detail=f"Todos los modelos están saturados ahora mismo: {str(ultimo_error)}"
    )
