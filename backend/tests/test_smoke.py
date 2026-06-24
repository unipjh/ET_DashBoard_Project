import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app


class BackendSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_imports_and_responds(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "ET API is running")

    @patch("backend.main.repo.check_db", return_value=(False, "db unavailable"))
    def test_health_reports_degraded_database(self, _check_db):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual(body["status"], "degraded")
        self.assertEqual(body["database"], "unavailable")

    @patch("backend.routers.articles.repo.get_articles_paginated", side_effect=RuntimeError("db unavailable"))
    def test_article_list_reports_database_unavailable(self, _articles):
        response = self.client.get("/api/articles?page=1&size=10")
        self.assertEqual(response.status_code, 503)

    @patch("backend.routers.admin.get_admin_password", return_value="secret")
    def test_admin_stats_requires_password(self, _password):
        response = self.client.get("/api/admin/stats")
        self.assertEqual(response.status_code, 401)

    @patch("backend.routers.admin.get_admin_password", return_value="")
    def test_admin_stats_reports_missing_server_password(self, _password):
        response = self.client.get("/api/admin/stats")
        self.assertEqual(response.status_code, 503)

    @patch("backend.routers.admin.repo.load_articles_for_stats", return_value=pd.DataFrame())
    @patch("backend.routers.admin.get_admin_password", return_value="secret")
    def test_admin_stats_accepts_valid_password(self, _password, _load_articles):
        response = self.client.get(
            "/api/admin/stats",
            headers={"X-Admin-Password": "secret"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_articles"], 0)

    @patch("backend.routers.admin.repo.load_articles_for_stats", side_effect=RuntimeError("db unavailable"))
    @patch("backend.routers.admin.get_admin_password", return_value="secret")
    def test_admin_stats_reports_database_unavailable(self, _password, _load_articles):
        response = self.client.get(
            "/api/admin/stats",
            headers={"X-Admin-Password": "secret"},
        )
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
