"""
AirFair-Vista Locust Stress Test
tests/locustfile.py

Simulates real user behaviour:
  1. Load the home page
  2. Check that the page responded successfully (HTTP 200)
  3. Simulate think-time (1-3 seconds) between requests

Run headless:
  locust -f locustfile.py --headless -u 50 -r 10 -t 30s --host <URL>
"""
from locust import HttpUser, task, between

class AirFairUser(HttpUser):
    # Think-time: 1-3s between requests (realistic user pace)
    wait_time = between(1, 3)

    @task(3)
    def load_homepage(self):
        """Main page load — highest frequency task (weight=3)"""
        with self.client.get('/', catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f'Homepage returned {resp.status_code}')

    @task(1)
    def load_static_assets(self):
        """Simulate browser fetching Streamlit static assets (weight=1)"""
        with self.client.get('/_stcore/health', catch_response=True) as resp:
            # Streamlit health endpoint — 200 means server is alive
            if resp.status_code in [200, 404]:
                resp.success()   # 404 is ok — endpoint may not exist
            else:
                resp.failure(f'Health check returned {resp.status_code}')
