# import time
# import smtplib
# import MIMEMultipart, MIMEText

# # Initialize Notification System
# smtp_config = {
#     "server": "smtp.example.com", 
#     "port": 587, 
#     "email": "your_email@example.com",  # Sender email address
#     "password": "your_email_password",  # Sender email password
# }  

# size_limit = 1 * 1024 * 1024  # Set maximum message size to 1MB

# # Format Notification Content
# timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) 
# device_id = "Device1234"  # unique identifier of the edge device
# auxiliary_data = {"Battery Level": "85%", "Signal Strength": "Strong"}  

# # Construct the notification message
# subject = "Glass Break Detected"  # subject of the email
# message_body = (
#     f"Alert: A glass break event was detected.\n"
#     f"Time: {timestamp}\n"
#     f"Device ID: {device_id}\n"
#     f"Auxiliary Info: {auxiliary_data}\n"
# ) 

# # Ensure message size does not exceed 1MB
# if len(message_body.encode('utf-8')) > size_limit:
#     raise ValueError("Notification size exceeds 1MB limit.")  # Validate message size

# # Send Email Notification
# try:
#     # Connect to the SMTP server
#     with smtplib.SMTP(smtp_config['server'], smtp_config['port']) as server:
#         server.starttls()  # Enable TLS encryption
#         server.login(smtp_config['email'], smtp_config['password'])  # Authenticate sender email

#         # Create the email message
#         msg = MIMEMultipart()  # Initialize a multipart message
#         msg["From"] = smtp_config['email']  # Specify the sender
#         msg["To"] = "recipient_email@example.com"  # Specify the recipient email
#         msg["Subject"] = subject
#         msg.attach(MIMEText(message_body, "plain"))  # Attach the message body

#         # Send the email
#         server.send_message(msg)  # Send the email message
#         print(f"Email sent to recipient_email@example.com")  # Confirm email sent
# except Exception as e:
#     print(f"Failed to send email: {e}") 