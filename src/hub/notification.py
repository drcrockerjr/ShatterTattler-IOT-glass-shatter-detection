# from sinch import SinchClient
# import notification_keys

# # sinch_client = SinchClient(
# #     key_id=notification_keys.sinch_key_id,
# #     key_secret=notification_keys.sinch_key_secret,
# #     project_id=notification_keys.sinch_project_id
# # )

# glass_break_msg = "A glass break has been detected"


# def notifiy_glass_break(device_id, timestamp):
#     sinch_client = SinchClient(
#         key_id=notification_keys.sinch_key_id,
#         key_secret=notification_keys.sinch_key_secret,
#         project_id=notification_keys.sinch_project_id
#     )

#     glass_break_msg = f"A glass break has been detected at device {device_id} at {timestamp} !!"


#     send_batch_response = sinch_client.sms.batches.send(
#         body=glass_break_msg,
#         to=["9713128722"],
#         from_="2085813084",
#         delivery_report="none"
#     )

#     print(send_batch_response)

# send_batch_response = sinch_client.sms.batches.send(
#     body="Hello from Sinch!",
#     to=["9713128722"],
#     from_="2085813084",
#     delivery_report="none"
# )


from enum import Enum
import notification_keys
import smtplib
from typing import Optional
from email.message import EmailMessage

class AlertCode(Enum):
    GLASS_BREAK = 1
    POSSIBLE_GLASS_BREAK = 2
    NO_GLASS_BREAK = 3



def notify_user(alert_code:AlertCode, time_stamp:str, device_id:str):
    msg = EmailMessage()

    if alert_code == AlertCode.GLASS_BREAK:

        alert = f"Glass break has been detected from device {device_id} at {time_stamp}"
    elif alert_code == AlertCode.NO_GLASS_BREAK:
        alert = f"Sound was recorded, but NO glass break detected from device {device_id} as {time_stamp}"

    msg.set_content(alert)

    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = f'[SHATTER TATTLER] - Code: {alert_code}'
    msg['From'] = notification_keys.sender_email
    msg['To'] = notification_keys.recv_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(notification_keys.sender_email, notification_keys.sender_app_pass)
        server.sendmail(notification_keys.sender_email, notification_keys.recv_email, msg.as_string())
    # s.send_message(msg)
    # s.quit()

# notify_user(AlertCode.GLASS_BREAK, "4pm", "0440")
