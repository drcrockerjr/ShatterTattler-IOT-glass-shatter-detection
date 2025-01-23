from sinch import SinchClient

sinch_client = SinchClient(
    key_id="fb8787a6-d34d-4607-b4ed-bf10d71f90e1",
    key_secret="cU3uRImNKyS~F7XguMmjR~jeGK",
    project_id="57856f37-3032-481d-8044-444170d1d569"
)

send_batch_response = sinch_client.sms.batches.send(
    body="Hello from Sinch!",
    to=["9713128722"],
    from_="2085813084",
    delivery_report="none"
)

print(send_batch_response)