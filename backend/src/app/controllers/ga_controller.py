import asyncio
import json
import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..schemas.prediccion import (
    GAComparacionResponse,
    GAHistorialResponse,
    ReentrenarRequest,
    ReentrenarResponse,
)
from ..services.riesgo_materno_service import (
    esta_entrenando,
    obtener_comparacion_ga,
    obtener_historial_ga,
    reentrenar_ga,
    reentrenar_ga_con_progreso,
)

router = APIRouter(prefix="/api/v1/ga", tags=["Algoritmo genetico"])


@router.get("/estado")
def estado_ga():
    """Indica si hay un entrenamiento en curso en el servidor."""
    return {"en_entrenamiento": esta_entrenando()}


@router.get("/historial", response_model=GAHistorialResponse)
def historial_ga() -> GAHistorialResponse:
    return GAHistorialResponse(**obtener_historial_ga())


@router.get("/comparacion", response_model=GAComparacionResponse)
def comparacion_ga() -> GAComparacionResponse:
    return GAComparacionResponse(**obtener_comparacion_ga())


@router.post("/reentrenar", response_model=ReentrenarResponse)
def reentrenar(payload: ReentrenarRequest) -> ReentrenarResponse:
    """Lanza el entrenamiento genetico con los parametros recibidos. Puede tomar varios minutos."""
    try:
        resultado = reentrenar_ga(parametros=payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ReentrenarResponse(**resultado)


@router.post("/reentrenar-stream")
async def reentrenar_stream(payload: ReentrenarRequest, request: Request):
    """Streaming SSE del progreso de reentrenamiento generacion por generacion."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress_callback(data: dict):
        asyncio.run_coroutine_threadsafe(queue.put(data), loop)

    def run():
        try:
            resultado = reentrenar_ga_con_progreso(
                parametros=payload.model_dump(),
                progress_callback=progress_callback,
            )
            asyncio.run_coroutine_threadsafe(
                queue.put({"tipo": "done", **resultado}), loop
            )
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put({"tipo": "error", "mensaje": str(exc)}), loop
            )

    threading.Thread(target=run, daemon=True).start()

    async def generate():
        while True:
            if await request.is_disconnected():
                break
            try:
                data = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(data)}\n\n"
            if data.get("tipo") in ("done", "error"):
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
