"""Exercise the real API lifecycle and the data boundary exposed to reviewers."""

from fastapi.testclient import TestClient

from labelops.api import create_app


def test_api_review_retrain_and_export(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json()["status"] == "ok"
        before = client.get("/summary").json()
        item = client.get("/queue", params={"limit": 1}).json()["items"][0]
        assert not {"label", "target", "ground_truth", "y"}.intersection(item)
        assert len(item["probabilities"]) == 10
        for reviewer in ("alice", "bob"):
            response = client.post(
                "/annotations",
                json={"sample_id": item["id"], "reviewer": reviewer, "label": 3},
            )
            assert response.status_code == 201
        assert len(client.get("/annotations").json()["items"]) == 2
        assert str(item["id"]) in client.get("/annotations/export").text
        after = client.post("/retrain").json()
        assert after["labeled_count"] == before["labeled_count"] + 1
        assert all(
            row["id"] != item["id"]
            for row in client.get("/queue", params={"limit": 50}).json()["items"]
        )


def test_api_rejects_invalid_inputs_and_non_pool_ids(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        for bad_label in (10, -1, True, "3"):
            response = client.post(
                "/annotations", json={"sample_id": 0, "reviewer": "alice", "label": bad_label}
            )
            assert response.status_code == 422
        assert client.get("/queue?limit=5000").status_code == 422
        assert (
            client.post(
                "/annotations", json={"sample_id": 999999, "reviewer": "alice", "label": 1}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/annotations", json={"sample_id": 0, "reviewer": "  ", "label": 1}
            ).status_code
            == 422
        )
