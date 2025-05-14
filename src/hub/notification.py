from enum import Enum
import notification_keys
import smtplib
from typing import Optional
from email.message import EmailMessage
import json

class AlertCode(Enum):
    GLASS_BREAK = 1
    POSSIBLE_GLASS_BREAK = 2
    NO_GLASS_BREAK = 3
    PERIODIC_REPORT = 4



def notify_user(alert_code:AlertCode, time_stamp:str, device_id:str, extra_info: dict = None):
    msg = EmailMessage()

    alert = None
    if alert_code == AlertCode.GLASS_BREAK:

        alert = f"Glass break has been detected from device {device_id} at {time_stamp}"
    elif alert_code == AlertCode.NO_GLASS_BREAK:
        alert = f"Sound was recorded, but NO glass break detected from device {device_id} as {time_stamp}"
    # elif alert_code == AlertCode.NO_GLASS_BREAK:


    payload = alert + "\n\n" + json.dumps(extra_info, indent=2)

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

def periodic_report(alarm_state, curr_time_stamp:str, prev_time_stamp,device_id:str, device_vbat, report_interval):
    msg = EmailMessage()
    report = (
            f"Periodic report for {device_id} @ {curr_time_stamp}\n\n"
            f"Number of Detected Glass Break detections since {prev_time_stamp} is {alarm_state['red']}\n\n "
            f"Number of Potential Glass Break detections since {prev_time_stamp} is {alarm_state['yellow']} \n\n"
            f"Edge Detection Device Battery Level (Voltage): {device_vbat}\n\n"
        )
    msg.set_content(report)

    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = f'[SHATTER TATTLER] - Periodic Report - {curr_time_stamp}'
    msg['From'] = notification_keys.sender_email
    msg['To'] = notification_keys.recv_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(notification_keys.sender_email, notification_keys.sender_app_pass)
        server.sendmail(notification_keys.sender_email, notification_keys.recv_email, msg.as_string())
    