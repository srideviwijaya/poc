from locust import HttpUser, task, TaskSet, between

class UserBehavior(TaskSet):
    @task(1)
    def send_post_request(self):
        # Define the URL endpoint
        url = "/v2/models/llamav2/infer"  # Adjust to match your endpoint

        # Define the payload with multiple inputs
        input={"inputs": [{"name":"prompt","datatype":"BYTES","shape":[1],"data":["I've been feeling very tired and have a sore throat for the past few days. Should I be worried?"]}]}

        # Send POST request
        with self.client.post(url, json=input, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")

    @task(2)
    def send_post_request2(self):
        # Define the URL endpoint
        url = "/v2/models/llamav2/infer"  # Adjust to match your endpoint

        # Define the payload with multiple inputs
        input={"inputs": [{"name":"prompt","datatype":"BYTES","shape":[1],"data":["Hi, what can I do for a headache?"]}]}

        # Send POST request
        with self.client.post(url, json=input, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")


class llamav2user(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)  # Simulate user wait time between requests

    # Set host to the URL of your application
    host = "http://localhost:8000"  # Adjust to match your application's URL