import cv2
import numpy as np
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_includes_both_artifacts():
    assert client.get('/health').json()['trained_models_exist'] is True

def test_invalid_image():
    assert client.post('/predict', files={'file': ('bad.png', b'invalid')}).status_code == 400

def test_small_image_rejected():
    _, encoded = cv2.imencode('.png', np.zeros((16, 16, 3), dtype=np.uint8))
    response = client.post('/predict', files={'file': ('small.png', encoded.tobytes())})
    assert response.status_code == 400

def test_included_models_predict():
    _, encoded = cv2.imencode('.png', np.zeros((128, 128, 3), dtype=np.uint8))
    response = client.post('/predict', files={'file': ('tile.png', encoded.tobytes())})
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result['cnn_count'], int)
    assert len(result['tiles']) == 1
