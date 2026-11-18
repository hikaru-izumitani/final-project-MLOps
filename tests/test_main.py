from fastapi.testclient import TestClient
from final_project_mlops.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {"Hello": "World"}