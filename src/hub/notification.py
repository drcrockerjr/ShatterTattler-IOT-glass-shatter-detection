from enum import Enum
import notification_keys
import smtplib
from typing import Optional
from email.message import EmailMessage

class AlertCode(Enum):
    # Different types of alerts for glass-break detection
    GLASS_BREAK = 1
    POSSIBLE_GLASS_BREAK = 2
    NO_GLASS_BREAK = 3

def notify_user(alert_code: AlertCode, time_stamp: str, device_id: str):
    """
    Send an email notification based on the provided alert code, timestamp, and device ID.
    """
    # Create a new email message object
    msg = EmailMessage()

    # Choose the email body content based on the alert code
    if alert_code == AlertCode.GLASS_BREAK:
        alert = (
            f"Glass break has been detected from device {device_id} "
            f"at {time_stamp}."
        )
    elif alert_code == AlertCode.NO_GLASS_BREAK:
        alert = (
            f"Sound was recorded, but NO glass break detected from device "
            f"{device_id} at {time_stamp}."
        )
    else:
        # Fallback for other alert codes
        alert = (
            f"Alert code {alert_code.name} received from device {device_id} "
            f"at {time_stamp}."
        )

    # Set the email body
    msg.set_content(alert)

    # Construct email headers
    msg['Subject'] = f'[SHATTER TATTLER] - Alert: {alert_code.name}'
    msg['From']    = notification_keys.sender_email   # Your Gmail address
    msg['To']      = notification_keys.recv_email     # Recipient address

    # Send the email over a secure SSL connection
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        # Log in using an app-specific password
        server.login(
            notification_keys.sender_email,
            notification_keys.sender_app_pass
        )
        # Send the message
        server.send_message(msg)