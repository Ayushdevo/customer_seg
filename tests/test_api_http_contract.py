from contextlib import contextmanager
import http.client
from http.server import HTTPServer
import json
import threading

import pytest

from test_vercel_api import endpoint


@contextmanager
def serving():
    class QuietHandler(endpoint.handler):
        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded"])
def test_http_rejects_non_json_without_running_prediction(monkeypatch, content_type):
    def must_not_predict(payload):
        raise AssertionError("Unsupported media types must not run prediction")

    monkeypatch.setattr(endpoint, "predict", must_not_predict)
    headers = {} if content_type is None else {"Content-Type": content_type}
    with serving() as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request("POST", "/api/predict", b"{}", headers)
        response = connection.getresponse()
        assert response.status == 415
        assert json.loads(response.read())["error"] == "Content-Type must be application/json"
        connection.close()


@pytest.mark.parametrize("content_type", ["application/json", "application/json; charset=utf-8", "Application/JSON"])
def test_http_accepts_json_media_type_and_charset(monkeypatch, content_type):
    monkeypatch.setattr(endpoint, "predict", lambda payload: {"cluster": 0})
    with serving() as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request("POST", "/api/predict", b"{}", {"Content-Type": content_type})
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-store"
        assert json.loads(response.read()) == {"cluster": 0}
        connection.close()
