from django.test import TestCase,Client
class DataFlowTests(TestCase):
 def test_health(self):
  r=Client().get("/api/v1/health/");self.assertEqual(r.status_code,200);self.assertEqual(r.json()["status"],"ok")
 def test_questions_contract(self):
  r=Client().get("/api/v1/diagnostic/questions?class=8&subject=maths");self.assertEqual(r.status_code,200);self.assertTrue(r.json()["questions"])
 def test_sync_contract(self):
  r=Client().post("/api/v1/sync/batch/",data='{"items":[{"id":"x","entity":"diagnostic_response","action":"create","payload":{}}]}',content_type="application/json");self.assertEqual(r.status_code,200);self.assertEqual(r.json()["accepted"],1)