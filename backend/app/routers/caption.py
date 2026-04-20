import logging
import os
import tempfile

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from app.schemas.caption import CaptionResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    loaded = hasattr(request.app.state, "caption_service") and request.app.state.caption_service is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@router.post("/predict", response_model=CaptionResponse)
async def predict(request: Request, image: UploadFile):
    service = getattr(request.app.state, "caption_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    _validate_file(image)

    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    suffix = os.path.splitext(image.filename or ".jpg")[1]
    tmp_path: str | None = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        tmp.write(contents)
        tmp.close()

        greedy_caption = service.generate(tmp_path, method="greedy")
        beam_caption = service.generate(tmp_path, method="beam", beam_width=3)

        return CaptionResponse(
            success=True,
            greedy_caption=greedy_caption,
            beam_caption=beam_caption,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed. Please try again.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
