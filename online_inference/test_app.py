from fastapi.testclient import TestClient
from .online_model import app, model


def test_predict():
    with TestClient(app) as client:
        response = client.get(
            "/predict/",
            json={"data": [
                [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589,
                 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622,
                 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
                [20.29, 14.34, 135.1, 1297.0, 0.1003, 0.1328, 0.198, 0.1043, 0.1809, 0.05883, 0.7572, 0.7813, 5.438,
                 94.44, 0.01149, 0.02461, 0.05688, 0.01885, 0.01756, 0.005115, 22.54, 16.67, 152.2, 1575.0, 0.1374,
                 0.205, 0.4, 0.1625, 0.2364, 0.07678],
                [11.42, 20.38, 77.58, 386.1, 0.1425, 0.2839, 0.2414, 0.1052, 0.2597, 0.09744, 0.4956, 1.156, 3.445,
                 27.23, 0.00911, 0.07458, 0.05661, 0.01867, 0.05963, 0.009208, 14.91, 26.5, 98.87, 567.7, 0.2098,
                 0.8663, 0.6869, 0.2575, 0.6638, 0.173]],
                "features": ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
                             'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
                             'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error',
                             'smoothness error', 'compactness error', 'concavity error', 'concave points error',
                             'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture',
                             'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness',
                             'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']

            }
        )
        assert response.status_code == 200
        assert response.json() == [{"condition": 0},
                                   {"condition": 0},
                                   {"condition": 0}
                                   ]