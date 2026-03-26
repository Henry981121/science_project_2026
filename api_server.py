"""
AI Detector — FastAPI Backend
==============================
啟動方式：
    uvicorn api_server:app --reload --port 8000
"""

import io
from typing import Optional
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from detector import AIDetector, STREAMS, STREAM_DISPLAY, conf_str, numpy_to_b64

app = FastAPI(title='AI Image Detector API', version='1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

print('Initializing AI Detector...')
detector = AIDetector()
print('API ready.')


@app.post('/predict')
async def predict(file: UploadFile = File(...), streams: Optional[str] = Form(None)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Please upload an image file.')

    contents = await file.read()
    try:
        img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='Cannot decode image.')

    enabled = [s for s in streams.split(',') if s in STREAMS] if streams else STREAMS
    if not enabled:
        raise HTTPException(status_code=400, detail='No valid streams selected.')

    result = detector.analyze_image(img_pil, enabled_streams=enabled)
    f = result['fusion']

    streams_out = {}
    for s in STREAMS:
        if s not in result['streams']:
            continue
        r = result['streams'][s]
        streams_out[s] = {
            'display_name': STREAM_DISPLAY[s],
            'prob_fake':    r['prob_fake'],
            'prediction':   r['prediction'],
            'weight':       result['stream_weights'].get(s, 0),
            'conf_str':     conf_str(r['prob_fake'], r['prediction']),
        }

    heatmap_b64 = None
    if 'resnet50' in result['heatmaps']:
        heatmap_b64 = numpy_to_b64(result['heatmaps']['resnet50'])

    return JSONResponse({
        'fusion': {
            'prob_fake':  f['prob_fake'],
            'prediction': f['prediction'],
            'conf_str':   conf_str(f['prob_fake'], f['prediction']),
        },
        'streams':      streams_out,
        'heatmap_b64':  heatmap_b64,
        'original_b64': numpy_to_b64(result['original']),
    })


@app.get('/health')
def health():
    return {'status': 'ok'}
